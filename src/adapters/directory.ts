/**
 * The peer directory: one view over every session this process can name.
 *
 * Discovery reuses `ctx.sessionQuery`, which already merges the live store with
 * the persistence backend and reports both availabilities. This adapter adds
 * only what that service cannot know — which *other host process* currently
 * holds a session — and turns the result into addressable peers.
 */

import type { AgentRegistry } from '@deepseek-ai/dsh-agent'
import type { SessionId } from '@deepseek-ai/dsh-session'
import type { SessionQueryService } from '@deepseek-ai/dsh-session-query'

import type { A2AEndpoint } from '../domain/a2a.ts'
import { assignPeerNames, type PeerDescriptor, type PeerLocation, type PeerNameSource } from '../domain/peer.ts'
import type { Logger, PeerDirectory } from '../ports/index.ts'
import type { PresenceStore } from './presence.ts'

/**
 * The listing fields a cached projection read uses as its identity witness.
 *
 * These are the authoritative header values a caller already holds from the store
 * listing: the cache matches them against its record and reads no log.
 */
export interface CachedTitleMeta {
  readonly version: number
  readonly id: SessionId
  readonly createdAt: number
  readonly isSeeded?: boolean
  readonly cwd?: string
}

/** One cached projection block: the wire values the checkpoint stored. */
export interface CachedTitleBlock {
  readonly values: Readonly<Record<string, unknown>>
}

/**
 * The durable projection cache's zero-I/O read faces.
 *
 * Declared structurally so a host without @deepseek-ai/dsh-session-projection-cache
 * needs no dependency, and a host with it passes the mounted service unchanged.
 */
export interface CachedTitleSource {
  cachedSnapshot(meta: CachedTitleMeta, keys?: readonly string[]): CachedTitleBlock | undefined
  cachedPredecessorTitle(meta: CachedTitleMeta): CachedTitleBlock | undefined
}

/** Construction inputs for {@link SessionQueryPeerDirectory}. */
export interface PeerDirectoryOptions {
  readonly sessionQuery: SessionQueryService
  readonly agents: AgentRegistry
  readonly presence: PresenceStore
  readonly logger: Logger
  /**
   * Reads the durable projection cache, when the deployment mounts one: the only
   * source of titles that reads no session log.
   *
   * A reader rather than the service, because Cordis activation is service-driven:
   * the cache's async init may still be pending when this plugin applies, and a
   * value captured then would stay undefined for the life of the process.
   */
  readonly projectionCache?: (() => CachedTitleSource | undefined) | undefined
  /** Whether sessions created as subagent children are addressable. */
  readonly includeSubagents: boolean
  /** External agents this deployment can reach, keyed by alias. */
  readonly a2aEndpoints?: Record<string, A2AEndpoint>
  /**
   * The aliases sessions have published about themselves, by session id.
   *
   * Supplied as a reader rather than a store so the directory keeps knowing
   * nothing about capability cards; without it, peers simply carry no alias.
   */
  readonly readAliases?: () => Promise<ReadonlyMap<string, string>>
}

/** A listing slower than this reports its own timing, because the caller cannot see where the time went. */
const SLOW_LISTING_MS = 1_000

/** Builds {@link PeerDescriptor}s from the harness session corpus. */
export class SessionQueryPeerDirectory implements PeerDirectory {
  readonly #sessionQuery: SessionQueryService
  readonly #agents: AgentRegistry
  readonly #presence: PresenceStore
  readonly #logger: Logger
  readonly #readProjectionCache: (() => CachedTitleSource | undefined) | undefined
  readonly #includeSubagents: boolean
  readonly #a2aEndpoints: Record<string, A2AEndpoint>
  readonly #readAliases: () => Promise<ReadonlyMap<string, string>>
  /** The listing in flight that concurrent callers share, cleared when it settles. */
  #listing: Promise<readonly PeerDescriptor[]> | undefined

  constructor(options: PeerDirectoryOptions) {
    this.#sessionQuery = options.sessionQuery
    this.#agents = options.agents
    this.#presence = options.presence
    this.#logger = options.logger
    this.#readProjectionCache = options.projectionCache
    this.#includeSubagents = options.includeSubagents
    this.#a2aEndpoints = options.a2aEndpoints ?? {}
    this.#readAliases = options.readAliases ?? (() => Promise.resolve(new Map()))
  }

  /**
   * List every addressable peer, newest session first.
   * @param signal - cancels this caller's wait; the shared walk itself still runs to completion.
   * @returns the peer set with names assigned and locations resolved.
   */
  async list(signal?: AbortSignal): Promise<readonly PeerDescriptor[]> {
    signal?.throwIfAborted()
    // Callers that ask while a walk runs share it (peer_list asks for its own
    // identity and its peer set together), so one store walk serves the burst.
    const shared = this.#sharedListing()
    if (signal === undefined) return shared
    // The walk itself is not cancellable, so it carries no signal: sharing it must
    // not let one caller's abort reject another caller whose signal is un-aborted.
    // Each caller races the shared walk against its own cancellation instead.
    return await new Promise<readonly PeerDescriptor[]>((resolve, reject) => {
      const onAbort = () => { reject(signal.reason ?? new Error('peer listing aborted')) }
      signal.addEventListener('abort', onAbort, { once: true })
      // Starting the walk can run a provider that aborts this caller synchronously,
      // before the listener attaches; an already-aborted signal never fires again.
      if (signal.aborted) {
        signal.removeEventListener('abort', onAbort)
        onAbort()
        return
      }
      shared.then((value) => {
        signal.removeEventListener('abort', onAbort)
        resolve(value)
      }, (error: unknown) => {
        signal.removeEventListener('abort', onAbort)
        reject(error)
      })
    })
  }

  /** The listing in flight, or a new shared walk; cleared when it settles. */
  #sharedListing(): Promise<readonly PeerDescriptor[]> {
    const current = this.#listing
    if (current !== undefined) return current
    const walk = this.#listOnce()
    this.#listing = walk
    const clear = () => { if (this.#listing === walk) this.#listing = undefined }
    walk.then(clear, clear)
    return walk
  }

  /**
   * Build one peer listing.
   *
   * Shared by concurrent callers for as long as the walk runs, so a caller that
   * arrives mid-walk sees the row set from when the walk started, not a fresher
   * one; that is the price of walking once per burst. A provider read here must
   * not call {@link list} itself, because it would await this very walk.
   */
  async #listOnce(): Promise<readonly PeerDescriptor[]> {
    const startedAt = Date.now()
    const [records, remoteHosts, aliases] = await Promise.all([
      this.#sessionQuery.listSessions(),
      this.#readPresence(),
      this.#aliases(),
    ])

    const visible = records.filter(
      (record) => this.#includeSubagents || record.header.origin !== 'subagent',
    )

    // Titles come from durable checkpoints when the host mounts the projection
    // cache: a log fold per session is a whole log decode, which is the entire
    // cost of listing peers on a large store.
    const titlesStartedAt = Date.now()
    // One resolve per listing: the report and the title read must agree on which
    // source served the titles.
    const cache = this.#readProjectionCache?.()
    const titles = await this.#readTitles(visible.map((record) => record.header), cache)

    const sources: PeerNameSource[] = visible.map((record) => ({
      sessionId: record.header.id,
      ...(titles.get(record.header.id) === undefined ? {} : { title: titles.get(record.header.id) as string }),
      ...(record.header.cwd === undefined ? {} : { cwd: record.header.cwd }),
    }))
    const names = assignPeerNames(sources)

    /** Session id to the host advertising it. */
    const remoteBySession = new Map<string, { hostId: string; socketPath: string }>()
    for (const host of remoteHosts) {
      for (const sessionId of host.sessions) {
        remoteBySession.set(sessionId, { hostId: host.hostId, socketPath: host.socketPath })
      }
    }

    // External agents are peers with no session behind them: they are always
    // reachable in principle, and never live in the corpus sense.
    const external: PeerDescriptor[] = Object.values(this.#a2aEndpoints).map((endpoint) => ({
      sessionId: `a2a:${endpoint.alias}`,
      name: endpoint.alias,
      createdAt: 0,
      live: true,
      location: { kind: 'a2a', alias: endpoint.alias },
    }))

    const sessions = visible.map((record) => {
      const sessionId = record.header.id
      const localAgent = this.#agents.get(sessionId as SessionId)
      const remote = remoteBySession.get(sessionId)
      const title = titles.get(sessionId)

      const location: PeerLocation = localAgent
        ? { kind: 'local' }
        : remote
          ? { kind: 'remote', hostId: remote.hostId, socketPath: remote.socketPath }
          : { kind: 'offline' }

      return {
        sessionId,
        name: names.get(sessionId) ?? sessionId,
        ...(aliases.get(sessionId) === undefined ? {} : { alias: aliases.get(sessionId) as string }),
        ...(title === undefined ? {} : { title }),
        ...(record.header.cwd === undefined ? {} : { cwd: record.header.cwd }),
        createdAt: record.header.createdAt,
        // `record.live` is this process's store; a remote host is live too.
        live: location.kind !== 'offline',
        ...(localAgent === undefined ? {} : { status: localAgent.status }),
        ...(record.header.origin === undefined ? {} : { origin: record.header.origin }),
        location,
      } satisfies PeerDescriptor
    })

    // A listing that takes seconds is a store walk, and the caller cannot see
    // which stage paid it: say so, with the counts that name the cause.
    const elapsed = Date.now() - startedAt
    if (elapsed >= SLOW_LISTING_MS) {
      this.#logger.warn(
        `peer listing took ${elapsed}ms over store listing, presence and aliases: ` +
          `${records.length} stored sessions, ${visible.length} listed, ` +
          `${titles.size} titles (${Date.now() - titlesStartedAt}ms), ` +
          `${cache !== undefined ? 'from the projection cache' : 'folded from logs'}`,
      )
    }
    return [...sessions, ...external]
  }

  /**
   * Read published aliases, tolerating an unavailable card store.
   * @returns session id to published alias, empty when unreadable.
   */
  async #aliases(): Promise<ReadonlyMap<string, string>> {
    try {
      return await this.#readAliases()
    } catch (error) {
      // Addressing by name and id must survive an unreadable card directory.
      this.#logger.warn(
        `capability cards unavailable: ${error instanceof Error ? error.message : String(error)}`,
      )
      return new Map()
    }
  }

  async #readPresence(): Promise<readonly { hostId: string; socketPath: string; sessions: readonly string[] }[]> {
    try {
      return await this.#presence.readPeers()
    } catch (error) {
      // Discovery of local sessions must survive an unreadable presence directory.
      this.#logger.warn(
        `presence unavailable: ${error instanceof Error ? error.message : String(error)}`,
      )
      return []
    }
  }

  /**
   * Read the latest title for each session without decoding a log when the
   * deployment mounts the projection cache; fold through the corpus otherwise.
   * @param headers - the listed sessions, whose headers are the cache's identity witness.
   * @param cache - the projection cache resolved once in this listing, or undefined to fold through the corpus.
   * @returns session id to title, omitting sessions without a title.
   */
  async #readTitles(
    headers: readonly CachedTitleMeta[],
    cache: CachedTitleSource | undefined,
  ): Promise<ReadonlyMap<string, string>> {
    const titles = new Map<string, string>()
    if (headers.length === 0) return titles

    if (cache !== undefined) {
      // A served block can carry a null title, and that must not hide the
      // predecessor face; chain on the title, not on the block. Both faces view
      // stored rows and neither opens a log.
      for (const header of headers) {
        const title = cachedTitle(cache.cachedSnapshot(header, ['title'])) ??
          cachedTitle(cache.cachedPredecessorTitle(header))
        if (title !== undefined) titles.set(header.id, title)
      }
      return titles
    }

    try {
      const observations = await this.#sessionQuery.readTitleSnapshots(
        headers.map((header) => header.id),
      )
      for (const observation of observations) {
        if (observation.status !== 'fulfilled') continue
        const title = observation.value.title?.title
        if (title) titles.set(observation.value.session.id, title)
      }
    } catch (error) {
      // Names fall back to directory or id; an unavailable index is not fatal.
      this.#logger.warn(
        `title folding unavailable: ${error instanceof Error ? error.message : String(error)}`,
      )
    }
    return titles
  }
}

/**
 * The title text a cached projection block carries, if any.
 *
 * The title projection's wire value is the latest logged text or null, so a block
 * that served it holds a non-empty string.
 * @param block - the viewed checkpoint block, or undefined when no row matched.
 * @returns the title text, or undefined when the block carries none.
 */
function cachedTitle(block: CachedTitleBlock | undefined): string | undefined {
  const value = block?.values.title
  return typeof value === 'string' && value !== '' ? value : undefined
}
