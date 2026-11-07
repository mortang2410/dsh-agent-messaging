/**
 * Peer titles come from durable projection checkpoints, not from log folds.
 *
 * Discovery lists every session in the store. Folding a title per id means one
 * whole-log decode per session, so on a large store a single listing call costs
 * minutes; the cache's zero-I/O faces exist precisely to avoid that.
 *
 * The cache matches a record to the caller's lifecycle through the listed
 * header's identity fields (`version`, `createdAt`, `cwd`, `isSeeded`), so the
 * fakes here match on exactly those, as the real cache does. The fakes restate
 * the field names, so they catch a header that does not carry them but not a
 * rename inside this plugin; the type and the identity test cover that.
 */
import { describe, expect, it, vi } from 'vitest'

import { SessionQueryPeerDirectory, type CachedTitleSource } from '../src/adapters/directory.ts'

/** A listed header, shaped as the store lists it. */
const header = (id: string) => ({
  version: 4,
  id,
  createdAt: 1,
  isSeeded: false,
  cwd: '/work',
})

const logger = { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} }

/** Match one witness the way the real cache's lifecycle identity does. */
function witnessMatches(meta: { version: number; createdAt: number; cwd?: string; isSeeded?: boolean }): boolean {
  return meta.version === 4 && meta.createdAt === 1 && meta.cwd === '/work' && (meta.isSeeded ?? false) === false
}

function directory(options: {
  cache?: CachedTitleSource
  foldCalls?: { count: number }
  seen?: { version: unknown }[]
}) {
  const foldCalls = options.foldCalls ?? { count: 0 }
  const sessionQuery = {
    listSessions: async () => [{ header: header('s1'), revision: 'r1' }, { header: header('s2'), revision: 'r2' }],
    readTitleSnapshots: vi.fn(async (ids: readonly string[]) => {
      foldCalls.count += 1
      return ids.map((id) => ({
        status: 'fulfilled' as const,
        value: { session: header(id), title: { title: `folded ${id}` } },
      }))
    }),
  }
  return new SessionQueryPeerDirectory({
    sessionQuery: sessionQuery as never,
    agents: { get: () => undefined } as never,
    presence: { readPeers: async () => [] } as never,
    logger: logger as never,
    includeSubagents: false,
    ...(options.cache === undefined ? {} : { projectionCache: () => options.cache }),
  })
}

/** The two faces answer with their own title, so a test can tell which one served. */
const caching = (
  snapshot: (meta: { id: string }) => string | undefined,
  predecessor: (meta: { id: string }) => string | undefined = () => undefined,
  calls = { snapshot: 0, predecessor: 0 },
): CachedTitleSource => ({
  cachedSnapshot: (meta, keys) => {
    calls.snapshot += 1
    const title = keys?.includes('title') === true && witnessMatches(meta) ? snapshot(meta) : undefined
    return title === undefined ? undefined : { values: { title } }
  },
  cachedPredecessorTitle: (meta) => {
    calls.predecessor += 1
    const title = witnessMatches(meta) ? predecessor(meta) : undefined
    return title === undefined ? undefined : { values: { title } }
  },
})

describe('peer titles', () => {
  it('reads titles from the projection cache and never folds a log', async () => {
    const foldCalls = { count: 0 }
    const calls = { snapshot: 0, predecessor: 0 }
    const cache = caching((meta) => `cached ${meta.id}`, (meta) => `stale ${meta.id}`, calls)
    const peers = await directory({ cache, foldCalls }).list()
    // The faces disagree on purpose: a snapshot hit must win, and consulting the
    // predecessor at all is a second lookup the hit already answered.
    expect(peers.map((peer) => peer.title)).toEqual(['cached s1', 'cached s2'])
    expect(calls.predecessor).toBe(0)
    expect(foldCalls.count).toBe(0)
  })

  it('passes the listed header itself as the identity witness', async () => {
    const seen: { version: unknown }[] = []
    const cache: CachedTitleSource = {
      cachedSnapshot: (meta) => {
        seen.push(meta)
        return undefined
      },
      cachedPredecessorTitle: () => undefined,
    }
    await directory({ cache }).list()
    expect(seen).toEqual([header('s1'), header('s2')])
  })

  it('serves a stale-record title from the predecessor checkpoint', async () => {
    const cache: CachedTitleSource = {
      cachedSnapshot: () => undefined,
      cachedPredecessorTitle: (meta) => ({ values: { title: `predecessor ${meta.id}` } }),
    }
    const peers = await directory({ cache }).list()
    expect(peers.map((peer) => peer.title)).toEqual(['predecessor s1', 'predecessor s2'])
  })

  it('omits a title the cache cannot serve instead of folding the log', async () => {
    const foldCalls = { count: 0 }
    const peers = await directory({ cache: caching(() => undefined), foldCalls }).list()
    expect(peers.every((peer) => peer.title === undefined)).toBe(true)
    expect(peers.map((peer) => peer.name)).toEqual(['work-s1', 'work-s2'])
    expect(foldCalls.count).toBe(0)
  })

  it('reports where a slow listing spent its time', async () => {
    vi.useFakeTimers()
    try {
      const warns: string[] = []
      const logger = { warn: (line: string) => warns.push(line), info: () => {}, error: () => {}, debug: () => {} }
      const sessionQuery = {
        listSessions: async () => {
          vi.advanceTimersByTime(2_000)
          return [{ header: header('s1'), revision: 'r1' }]
        },
        readTitleSnapshots: async () => [],
      }
      const cache = caching((meta) => `cached ${meta.id}`)
      const peers = new SessionQueryPeerDirectory({
        sessionQuery: sessionQuery as never,
        agents: { get: () => undefined } as never,
        presence: { readPeers: async () => [] } as never,
        logger: logger as never,
        includeSubagents: false,
        projectionCache: () => cache,
      })
      await peers.list()
      expect(warns).toHaveLength(1)
      expect(warns[0]).toContain('peer listing took 2000ms')
      expect(warns[0]).toContain('1 stored sessions')
      expect(warns[0]).toContain('from the projection cache')
    } finally {
      vi.useRealTimers()
    }
  })

  it('serves the title when the current block carries a null title but the predecessor has one', async () => {
    const cache: CachedTitleSource = {
      cachedSnapshot: () => ({ values: { title: null } }),
      cachedPredecessorTitle: (meta) => ({ values: { title: `predecessor ${meta.id}` } }),
    }
    const peers = await directory({ cache }).list()
    expect(peers.map((peer) => peer.title)).toEqual(['predecessor s1', 'predecessor s2'])
  })

  it('treats a null, an empty and a non-string title as no title', async () => {
    for (const served of [null, '', 42]) {
      const cache: CachedTitleSource = {
        cachedSnapshot: () => ({ values: { title: served } }),
        cachedPredecessorTitle: () => undefined,
      }
      const peers = await directory({ cache }).list()
      expect(peers.every((peer) => peer.title === undefined)).toBe(true)
    }
  })

  it('picks up a cache that finishes activating after this plugin applied', async () => {
    let cache: CachedTitleSource | undefined
    const foldCalls = { count: 0 }
    const sessionQuery = {
      listSessions: async () => [{ header: header('s1'), revision: 'r1' }],
      readTitleSnapshots: vi.fn(async (ids: readonly string[]) => {
        foldCalls.count += 1
        return ids.map((id) => ({ status: 'fulfilled' as const, value: { session: header(id), title: { title: `folded ${id}` } } }))
      }),
    }
    const peers = new SessionQueryPeerDirectory({
      sessionQuery: sessionQuery as never,
      agents: { get: () => undefined } as never,
      presence: { readPeers: async () => [] } as never,
      logger: { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} } as never,
      includeSubagents: false,
      projectionCache: () => cache,
    })
    expect((await peers.list()).map((peer) => peer.title)).toEqual(['folded s1'])
    cache = caching((meta) => `cached ${meta.id}`)
    expect((await peers.list()).map((peer) => peer.title)).toEqual(['cached s1'])
    expect(foldCalls.count).toBe(1)
  })

  it('walks the store once when two callers list in the same tick', async () => {
    const sessionQuery = {
      listSessions: vi.fn(async () => [{ header: header('s1'), revision: 'r1' }]),
      readTitleSnapshots: async () => [],
    }
    const peers = new SessionQueryPeerDirectory({
      sessionQuery: sessionQuery as never,
      agents: { get: () => undefined } as never,
      presence: { readPeers: async () => [] } as never,
      logger: { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} } as never,
      includeSubagents: false,
    })
    const [first, second] = await Promise.all([peers.list(), peers.list()])
    expect(first).toEqual(second)
    expect(sessionQuery.listSessions).toHaveBeenCalledTimes(1)
  })

  it('rejects only the caller that aborted, not a peer sharing the walk', async () => {
    let release: (() => void) | undefined
    const gate = new Promise<void>((resolve) => { release = resolve })
    const sessionQuery = {
      listSessions: vi.fn(async () => {
        await gate
        return [{ header: header('s1'), revision: 'r1' }]
      }),
      readTitleSnapshots: async () => [],
    }
    const peers = new SessionQueryPeerDirectory({
      sessionQuery: sessionQuery as never,
      agents: { get: () => undefined } as never,
      presence: { readPeers: async () => [] } as never,
      logger: { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} } as never,
      includeSubagents: false,
    })
    const first = new AbortController()
    const second = new AbortController()
    const one = peers.list(first.signal)
    const rejects = expect(one).rejects.toThrow('caller gave up')
    const two = peers.list(second.signal)
    first.abort(new Error('caller gave up'))
    await rejects
    release?.()
    // The walk itself is not cancelled, so the caller that never aborted still gets it.
    await expect(two).resolves.toHaveLength(1)
    expect(sessionQuery.listSessions).toHaveBeenCalledTimes(1)
  })

  it('rejects an already-aborted caller before walking the store', async () => {
    const sessionQuery = {
      listSessions: vi.fn(async () => [{ header: header('s1'), revision: 'r1' }]),
      readTitleSnapshots: async () => [],
    }
    const peers = new SessionQueryPeerDirectory({
      sessionQuery: sessionQuery as never,
      agents: { get: () => undefined } as never,
      presence: { readPeers: async () => [] } as never,
      logger: { warn: () => {}, info: () => {}, error: () => {}, debug: () => {} } as never,
      includeSubagents: false,
    })
    const controller = new AbortController()
    controller.abort(new Error('already gone'))
    await expect(peers.list(controller.signal)).rejects.toThrow('already gone')
    expect(sessionQuery.listSessions).not.toHaveBeenCalled()
  })

  it('folds through the corpus once when the host mounts no cache', async () => {
    const foldCalls = { count: 0 }
    const peers = await directory({ foldCalls }).list()
    expect(peers.map((peer) => peer.title)).toEqual(['folded s1', 'folded s2'])
    expect(foldCalls.count).toBe(1)
  })
})
