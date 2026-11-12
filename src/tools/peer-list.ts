/**
 * `peer_list` — which sessions this one can address.
 */

import { defineTool, type ToolDefinition } from '@deepseek-ai/dsh-tools'

import type { CardStore } from '../adapters/cards.ts'
import type { WorkspaceFacts } from '../domain/derived-card.ts'
import { deriveCard } from '../domain/derived-card.ts'
import type { TaskStateStore } from '../adapters/task-states.ts'
import type { WorkClaims } from '../app/claim-work.ts'
import type { MessageSender, SenderIdentity } from '../app/send-message.ts'
import { summarizeCard } from '../domain/card.ts'
import { describeAge, lastActivityAt, peerAddress, type PeerDescriptor } from '../domain/peer.ts'
import { summarizeTaskState } from '../domain/task-state.ts'
import { presentListCall } from './presentation.ts'
import { requireCallerSessionId } from './caller.ts'

/**
 * Reachability as the model should reason about it.
 *
 * A session that is not running carries its age, because the benchmark caught a
 * session deferring work to peers that had been dead for hours: it read their
 * titles, concluded the job was theirs, and left a call site broken. A stopped
 * session and a colleague looked identical in the listing, and a title is a
 * strong, unreliable signal.
 * @param peer - the peer to describe.
 * @param now - the current time, supplied so a listing is testable.
 * @returns the state, with an age for anything that is not running.
 */
function reachability(peer: PeerDescriptor, now: number): string {
  switch (peer.location.kind) {
    case 'local':
      return peer.status === 'running' ? 'running' : 'idle'
    case 'remote':
      return 'running (another host process)'
    case 'a2a':
      return 'external agent (A2A)'
    case 'offline':
      return `not running, started ${describeAge(Math.max(0, now - peer.createdAt))}`
  }
}

/** One row of a `peer_list` result: the calling session or one peer. */
interface ListedSession {
  readonly name: string
  readonly session_id: string
  readonly state: string
  readonly is_self: boolean
  readonly title?: string
  readonly cwd?: string
  readonly task?: string
  readonly role?: string
  working_on?: string[]
  hidden_peers?: HiddenPeers
}

/** What a listing left out, so a reader does not mistake the view for the store. */
interface HiddenPeers {
  readonly total_peers: number
  readonly older_than_days: number
  readonly beyond_cap: number
}

/** How far back a peer may have last acted and still be listed by default. */
const DEFAULT_ACTIVE_DAYS = 3
/** How many sessions one listing returns by default, the caller's own row included. */
const DEFAULT_MAX_SESSIONS = 30
const DAY_MS = 24 * 60 * 60 * 1000

/**
 * A caller-supplied bound, made usable.
 * @param value - the argument as the model sent it, if it sent one.
 * @param fallback - the bound this listing applies when the argument is not a bound.
 * @returns the value when it is a non-negative finite number, 0 included; the
 *   fallback otherwise, because a negative or broken argument is not a request
 *   for a view, and 0 already means "no limit".
 */
function limit(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value) || value < 0) return fallback
  return value
}

/**
 * How many sessions a listing may return.
 * @param maxSessions - the bounded argument, 0 for no cap.
 * @returns 0 for no cap, otherwise a whole number of sessions, at least the caller's own row.
 */
function sessionCap(maxSessions: number): number {
  return maxSessions === 0 ? 0 : Math.max(1, Math.floor(maxSessions))
}

/** The peers one listing shows, and the ones each rule leaves out. */
interface PeerSelection {
  readonly listed: readonly PeerDescriptor[]
  readonly olderThanWindow: number
  readonly beyondCap: number
}

/**
 * Choose the peers one listing shows.
 *
 * A live peer enters the view whatever its age: it is the colleague the caller
 * is addressing. An offline one enters only while it has acted inside the
 * window, because the store keeps every session ever created and a listing of
 * all of them stops being readable long before it stops growing. The cap can
 * still drop either kind, newest activity first.
 * @param peers - every addressable peer except the caller.
 * @param activeDays - age of the oldest activity still listed; 0 for no window.
 * @param maxSessions - most sessions to return, the caller's own row included; 0 for no cap.
 * @param now - the current time, supplied so a listing is testable.
 * @returns the listed peers, newest activity first, and the two omission counts.
 */
function selectPeers(
  peers: readonly PeerDescriptor[],
  activeDays: number,
  maxSessions: number,
  now: number,
): PeerSelection {
  const windowStart = activeDays > 0 ? now - activeDays * DAY_MS : Number.NEGATIVE_INFINITY
  const reachable = peers.filter((peer) => peer.live || lastActivityAt(peer) >= windowStart)
  // One rule ranks the view: last activity, because that is what the cap keeps
  // when it cuts. Reachability only breaks a tie, so a live peer cannot outrank
  // a peer the caller talked to an hour ago.
  const ordered = reachable.toSorted(
    (left, right) => lastActivityAt(right) - lastActivityAt(left) || Number(right.live) - Number(left.live),
  )
  // The caller's own row is always first and always listed, so it spends one
  // place of the cap: 30 sessions means 30 entries, not 30 peers beside SELF.
  const cap = sessionCap(maxSessions)
  const listed = cap === 0 ? ordered : ordered.slice(0, cap - 1)
  return {
    listed,
    olderThanWindow: peers.length - reachable.length,
    beyondCap: ordered.length - listed.length,
  }
}

/**
 * The line that keeps a short listing honest about the sessions it omits.
 * @param hidden - what the listing left out.
 * @param activeDays - the window this listing applied.
 * @param maxSessions - the cap this listing applied.
 * @returns one sentence naming the omissions and how to widen the view.
 */
function renderOmission(hidden: HiddenPeers, activeDays: number, maxSessions: number): string {
  const reasons: string[] = []
  const window = `${activeDays} ${activeDays === 1 ? 'day' : 'days'}`
  if (hidden.older_than_days > 0) reasons.push(`${hidden.older_than_days} idle longer than ${window}`)
  if (hidden.beyond_cap > 0) reasons.push(`${hidden.beyond_cap} beyond the ${sessionCap(maxSessions)}-session limit`)
  const shown = hidden.total_peers - hidden.older_than_days - hidden.beyond_cap
  return (
    `${hidden.total_peers} peers; ${shown} listed, ${reasons.join(' and ')} not listed. ` +
    'Pass active_days or max_sessions to widen this, or address any session by its full id.'
  )
}

/**
 * Render one `peer_list` row as the block a model reads.
 * @param entry - the row to render.
 * @returns a SELF or PEER block with the full session id on its own line.
 */
function renderEntry(entry: ListedSession): string {
  const label = entry.is_self ? 'SELF' : 'PEER'
  const title = entry.title ? ` "${entry.title}"` : ''
  const lines = [`${label}: ${entry.name} [${entry.state}]${title}`, `  session_id: ${entry.session_id}`]
  if (entry.cwd) lines.push(`  cwd: ${entry.cwd}`)
  if (entry.task) lines.push(`  task: ${entry.task}`)
  if (entry.role) lines.push(`  ${entry.role}`)
  if (entry.working_on && entry.working_on.length > 0) {
    lines.push(`  working on: ${entry.working_on.join(', ')}`)
  }
  return lines.join('\n')
}

/**
 * Build the `peer_list` tool.
 * @param sender - the send use case, which also owns discovery.
 * @param claims - live work claims, so a caller sees who is on what.
 * @param cards - capability cards, so a caller sees what each peer is for.
 * @param states - declared task states, so a caller sees what each peer's work is doing.
 * @param identify - resolves the calling session's own name and directory, from its authoritative id.
 * @param describe - reads a workspace for an honest card where none was published.
 * @returns the registry-ready definition.
 */
export function createPeerListTool(
  sender: MessageSender,
  claims: WorkClaims,
  cards: CardStore,
  states: TaskStateStore,
  identify: (sessionId: string, signal?: AbortSignal) => Promise<SenderIdentity>,
  describe?: (cwd: string) => Promise<Omit<WorkspaceFacts, 'sessionId'>>,
): ToolDefinition {
  return defineTool({
    name: 'peer_list',
    description: [
      'List the agent sessions you can address with peer_send. The first entry, marked SELF, is this session:',
      'its session_id is your own canonical ID, and other sessions see you under that same ID as a PEER.',
      'You cannot send to SELF. Each entry shows the full session_id. Use this listing when you need the exact',
      'session ID of a peer, or when an address was ambiguous.',
      'Sessions marked "not running" can still be sent to:',
      'the message is queued and delivered when that session next starts.',
      'It lists the sessions that are running plus those active in the last 3 days, at most 30 sessions',
      'counting this SELF entry; when more qualify than fit, the most recent activity wins.',
      'active_days and max_sessions widen that view, and 0 means no limit.',
      'This returns only session identities — never their conversation contents.',
      'A session marked "not running" is not working on anything: do not conclude that a task belongs to one',
      'because its title mentions that task. If work needs doing and nobody live owns it, it is yours or your user\'s.',
    ].join(' '),
    parameters: {
      active_days: {
        type: 'number',
        default: DEFAULT_ACTIVE_DAYS,
        description:
          `How many days back a session may have last been prompted and still be listed. A session a live agent holds is listed whatever its age, though max_sessions can still drop it. Default ${DEFAULT_ACTIVE_DAYS}; 0 removes the age limit.`,
      },
      max_sessions: {
        type: 'number',
        default: DEFAULT_MAX_SESSIONS,
        description:
          `The most sessions to list, counting the SELF entry, so the default ${DEFAULT_MAX_SESSIONS} means SELF plus ${DEFAULT_MAX_SESSIONS - 1} peers. 0 removes the limit.`,
      },
    },
    output: {
      schema: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            name: { type: 'string', required: true, description: 'The address to pass to peer_send.' },
            session_id: {
              type: 'string',
              required: true,
              description: 'The full, canonical session ID. Exact and unambiguous, unlike a name.',
            },
            state: { type: 'string', required: true, description: 'running, idle, or not running.' },
            is_self: {
              type: 'boolean',
              required: true,
              description: 'True for the calling session itself, which is always the first entry. It cannot be messaged.',
            },
            title: { type: 'string' },
            cwd: { type: 'string' },
            role: {
              type: 'string',
              description: "What this peer says it is for, and what it owns. Its own words, not a guess from its title.",
            },
            task: {
              type: 'string',
              description:
                "What this peer's work is doing, in its own words: working, blocked, done or abandoned. More informative than the process state.",
            },
            working_on: {
              type: 'array',
              description: 'Resources this peer has claimed. Do not edit these without talking to them.',
              items: { type: 'string' },
            },
            hidden_peers: {
              type: 'object',
              additionalProperties: false,
              description:
                'SELF only, and only when the view is short: how many peers exist and how many each rule left out. The listed rows are a window on the store, not the whole of it.',
              properties: {
                total_peers: {
                  type: 'integer',
                  required: true,
                  description: 'Sessions this caller can address, SELF excluded.',
                },
                older_than_days: {
                  type: 'integer',
                  required: true,
                  description: 'Peers left out for having no activity inside the age window.',
                },
                beyond_cap: {
                  type: 'integer',
                  required: true,
                  description: 'Peers left out by the row limit.',
                },
              },
            },
          },
        },
      },
      render: (args, value) => {
        if (value.length === 0) return [{ type: 'text', text: 'No sessions are reachable.' }]
        const blocks = value.map(renderEntry)
        const hidden = value.find((entry) => entry.is_self)?.hidden_peers
        if (hidden === undefined) {
          // The calling session is always listed, so "no peers" is its only other case.
          if (value.every((entry) => entry.is_self)) blocks.push('No other sessions are reachable.')
        } else {
          blocks.push(
            renderOmission(
              hidden,
              limit(args.active_days, DEFAULT_ACTIVE_DAYS),
              limit(args.max_sessions, DEFAULT_MAX_SESSIONS),
            ),
          )
        }
        return [{ type: 'text', text: blocks.join('\n\n') }]
      },
    },
    presentCall: () => presentListCall(),
    async execute(args, exec) {
      // The caller is read from the execution, never from arguments: a model
      // cannot list itself as someone else, or ask for a SELF it does not own.
      const self = requireCallerSessionId(exec)
      const [identity, peers] = await Promise.all([
        identify(self, exec.signal),
        sender.peers(self, exec.signal),
      ])
      // One clock serves the window and every age this listing prints.
      const now = Date.now()
      const selection = selectPeers(
        peers,
        limit(args.active_days, DEFAULT_ACTIVE_DAYS),
        limit(args.max_sessions, DEFAULT_MAX_SESSIONS),
        now,
      )
      const [held, published, declared] = await Promise.all([claims.all(), cards.readAll(), states.readAll()])
      const cardBySession = new Map(published.map((card) => [card.sessionId, card]))

      // A session is useful to its peers before it declares anything: where a
      // card was never published, the workspace is read for the honest part of
      // one, marked as inferred so nobody mistakes it for a statement. Only a
      // listed peer pays for it: one directory read per session is the costly half.
      if (describe !== undefined) {
        const candidates = [
          { sessionId: self, cwd: identity.cwd },
          ...selection.listed.map((peer) => ({ sessionId: peer.sessionId, cwd: peer.cwd })),
        ]
        await Promise.all(
          candidates
            .filter((entry) => !cardBySession.has(entry.sessionId) && entry.cwd !== undefined)
            .map(async (entry) => {
              try {
                const facts = await describe(entry.cwd as string)
                const card = deriveCard({ sessionId: entry.sessionId, ...facts }, now)
                if (card !== undefined) cardBySession.set(entry.sessionId, card)
              } catch {
                // A listing must survive an unreadable directory.
              }
            }),
        )
      }
      // A wait is recorded as a session id; a reader needs the address. The
      // calling session is included, so a peer waiting on it reads its name.
      const addressBySession = new Map<string, string>([
        [self, identity.name],
        ...peers.map((peer): [string, string] => [peer.sessionId, peerAddress(peer)]),
      ])
      const stateBySession = new Map(declared.map((state) => [state.sessionId, state]))

      /** Claimed resources by holding session, so a caller sees who is on what. */
      const byHolder = new Map<string, string[]>()
      for (const claim of held) {
        const list = byHolder.get(claim.sessionId) ?? []
        list.push(`${claim.resource} (${claim.intent})`)
        byHolder.set(claim.sessionId, list)
      }

      /** The role, task and claims a session has, shared by the SELF and PEER rows. */
      const metadataFor = (sessionId: string) => {
        const card = cardBySession.get(sessionId)
        const task = stateBySession.get(sessionId)
        const working = byHolder.get(sessionId) ?? []
        return {
          ...(task === undefined
            ? {}
            : { task: summarizeTaskState(task, (id) => addressBySession.get(id)) }),
          ...(card === undefined ? {} : { role: summarizeCard(card) }),
          ...(working.length === 0 ? {} : { working_on: working }),
        }
      }

      const omitted = selection.olderThanWindow + selection.beyondCap
      const selfEntry: ListedSession = {
        name: identity.name,
        session_id: self,
        // The caller is, by definition, running: it is the session making this call.
        state: 'running',
        is_self: true,
        ...(identity.cwd === undefined ? {} : { cwd: identity.cwd }),
        ...metadataFor(self),
        ...(omitted === 0
          ? {}
          : {
              hidden_peers: {
                total_peers: peers.length,
                older_than_days: selection.olderThanWindow,
                beyond_cap: selection.beyondCap,
              },
            }),
      }

      const peerEntries = selection.listed.map(
        (peer): ListedSession => ({
          name: peerAddress(peer),
          session_id: peer.sessionId,
          state: reachability(peer, now),
          is_self: false,
          ...(peer.title === undefined ? {} : { title: peer.title }),
          ...(peer.cwd === undefined ? {} : { cwd: peer.cwd }),
          ...metadataFor(peer.sessionId),
        }),
      )

      return [selfEntry, ...peerEntries]
    },
  })
}
