/**
 * The send path end to end: MessageSender → RoutingTransport → InboundRouter
 * → sink, with only the harness edges faked.
 */

import { describe, expect, it } from 'vitest'

import { InboundRouter } from '../src/app/receive-message.ts'
import { MessageSender, type SenderIdentity } from '../src/app/send-message.ts'
import { RoutingTransport } from '../src/adapters/transport/routing-transport.ts'
import type { Envelope } from '../src/domain/envelope.ts'
import { PeerError } from '../src/domain/errors.ts'
import { LoopGuard, type InboundPolicy } from '../src/domain/policy.ts'
import type { PeerDescriptor } from '../src/domain/peer.ts'
import type {
  DeliveryReceipt,
  InboxSink,
  OutboxSpool,
  PeerDirectory,
} from '../src/ports/index.ts'
import type { InboxClient } from '../src/adapters/transport/inbox-client.ts'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import type { CardStore } from '../src/adapters/cards.ts'
import type { TaskStateStore } from '../src/adapters/task-states.ts'
import type { WorkClaims } from '../src/app/claim-work.ts'
import { createIdentityResolver } from '../src/app/identity.ts'
import { createCard, type CapabilityCard } from '../src/domain/card.ts'
import { createTaskState, type TaskState } from '../src/domain/task-state.ts'
import { createPeerListTool } from '../src/tools/peer-list.ts'

const alice: SenderIdentity = { sessionId: 'session-a', name: 'alice', cwd: '/repo/a' }

function peer(overrides: Partial<PeerDescriptor> & Pick<PeerDescriptor, 'sessionId' | 'name'>): PeerDescriptor {
  return { createdAt: 0, live: true, location: { kind: 'local' }, ...overrides }
}

class FakeDirectory implements PeerDirectory {
  constructor(private readonly peers: readonly PeerDescriptor[]) {}
  list(): Promise<readonly PeerDescriptor[]> {
    return Promise.resolve(this.peers)
  }
}

class RecordingSink implements InboxSink {
  readonly delivered: Envelope[] = []
  deliver(envelope: Envelope): DeliveryReceipt {
    this.delivered.push(envelope)
    return { status: 'delivered' }
  }
}

class RecordingSpool implements OutboxSpool {
  readonly held: Envelope[] = []
  hold(envelope: Envelope): Promise<void> {
    this.held.push(envelope)
    return Promise.resolve()
  }
  drain(): Promise<readonly Envelope[]> {
    return Promise.resolve([])
  }
}

/** Stands in for a socket to another host. */
class RecordingClient {
  readonly sent: { socketPath: string; payload: unknown }[] = []
  receipt: DeliveryReceipt = { status: 'delivered' }
  send(socketPath: string, payload: unknown): Promise<DeliveryReceipt> {
    this.sent.push({ socketPath, payload })
    return Promise.resolve(this.receipt)
  }
}

interface Harness {
  sender: MessageSender
  sink: RecordingSink
  spool: RecordingSpool
  client: RecordingClient
  inbound: InboundRouter
}

function harness(
  peers: readonly PeerDescriptor[],
  options: { policy?: InboundPolicy; spoolOffline?: boolean } = {},
): Harness {
  const sink = new RecordingSink()
  const spool = new RecordingSpool()
  const client = new RecordingClient()

  const inbound = new InboundRouter({
    policy: options.policy ?? 'accept',
    guard: new LoopGuard({ maxPerWindow: 10, windowMs: 1_000, duplicateWindowMs: 100 }),
    sink,
    clock: { now: () => 0 },
    maxHeld: 10,
  })

  const sender = new MessageSender({
    directory: new FakeDirectory(peers),
    transport: new RoutingTransport({
      inbound,
      client: client as unknown as InboxClient,
      spool,
      spoolOffline: options.spoolOffline ?? true,
    }),
    clock: { now: () => 1_700_000_000_000 },
    ids: (() => {
      let n = 0
      return { next: () => `msg-${(n += 1)}` }
    })(),
  })

  return { sender, sink, spool, client, inbound }
}

describe('send pipeline', () => {
  const bob = peer({ sessionId: 'session-b', name: 'bob', cwd: '/repo/b' })

  it('delivers to a local peer', async () => {
    const h = harness([bob])
    const outcome = await h.sender.send(alice, { to: 'bob', body: 'rebase is safe', mode: 'followup' })

    expect(outcome.receipt.status).toBe('delivered')
    expect(h.sink.delivered).toHaveLength(1)
    expect(h.sink.delivered[0]?.body).toBe('rebase is safe')
  })

  it('stamps the sender from the caller identity, not from the request', async () => {
    const h = harness([bob])
    await h.sender.send(alice, { to: 'bob', body: 'hi', mode: 'followup' })
    expect(h.sink.delivered[0]?.from).toEqual({
      sessionId: 'session-a',
      name: 'alice',
      cwd: '/repo/a',
    })
  })

  it('returns a correlatable message id', async () => {
    const h = harness([bob])
    const outcome = await h.sender.send(alice, { to: 'bob', body: 'hi', mode: 'followup' })
    expect(outcome.envelope.id).toBe('msg-1')
  })

  it('carries the chosen delivery mode through to the receiver', async () => {
    const h = harness([bob])
    await h.sender.send(alice, { to: 'bob', body: 'stop', mode: 'steer' })
    expect(h.sink.delivered[0]?.mode).toBe('steer')
  })

  it('routes to another host over the transport client', async () => {
    const remote = peer({
      sessionId: 'session-r',
      name: 'remote',
      location: { kind: 'remote', hostId: 'h2', socketPath: '/tmp/h2.sock' },
    })
    const h = harness([remote])
    const outcome = await h.sender.send(alice, { to: 'remote', body: 'hello', mode: 'followup' })

    expect(outcome.receipt.status).toBe('delivered')
    expect(h.client.sent).toHaveLength(1)
    expect(h.client.sent[0]?.socketPath).toBe('/tmp/h2.sock')
    expect(h.sink.delivered).toHaveLength(0)
  })

  it('spools for a session that is not running', async () => {
    const offline = peer({ sessionId: 'session-o', name: 'offline', live: false, location: { kind: 'offline' } })
    const h = harness([offline])
    const outcome = await h.sender.send(alice, { to: 'offline', body: 'later', mode: 'followup' })

    expect(outcome.receipt.status).toBe('spooled')
    expect(h.spool.held.map((e) => e.body)).toEqual(['later'])
  })

  it('rejects an offline target when spooling is disabled', async () => {
    const offline = peer({ sessionId: 'session-o', name: 'offline', live: false, location: { kind: 'offline' } })
    const h = harness([offline], { spoolOffline: false })

    await expect(
      h.sender.send(alice, { to: 'offline', body: 'later', mode: 'followup' }),
    ).rejects.toThrowError(/not running/)
  })

  it('applies the receiver policy to a same-process delivery', async () => {
    // The guarantee: sharing a process with the recipient is not a way around
    // its inbound policy.
    const h = harness([bob], { policy: 'refuse' })
    const outcome = await h.sender.send(alice, { to: 'bob', body: 'hi', mode: 'followup' })

    expect(outcome.receipt.status).toBe('refused')
    expect(h.sink.delivered).toHaveLength(0)
  })

  it('applies loop control to a same-process delivery', async () => {
    const h = harness([bob])
    const first = await h.sender.send(alice, { to: 'bob', body: 'same', mode: 'followup' })
    const second = await h.sender.send(alice, { to: 'bob', body: 'same', mode: 'followup' })

    expect(first.receipt.status).toBe('delivered')
    expect(second.receipt.status).toBe('dropped')
    expect(h.sink.delivered).toHaveLength(1)
  })

  it('refuses to address the calling session, even by its own id', async () => {
    const self = peer({ sessionId: 'session-a', name: 'alice' })
    const h = harness([self, bob])
    await expect(
      h.sender.send(alice, { to: 'session-a', body: 'hi', mode: 'followup' }),
    ).rejects.toThrow(PeerError)
  })

  it('excludes the calling session from its own peer listing', async () => {
    const self = peer({ sessionId: 'session-a', name: 'alice' })
    const h = harness([self, bob])
    expect((await h.sender.peers('session-a')).map((p) => p.name)).toEqual(['bob'])
  })

  it('reports an unresolvable address', async () => {
    const h = harness([bob])
    await expect(
      h.sender.send(alice, { to: 'nobody', body: 'hi', mode: 'followup' }),
    ).rejects.toThrowError(/No session matches/)
  })

  it('preserves reply correlation end to end', async () => {
    const h = harness([bob])
    await h.sender.send(alice, { to: 'bob', body: 'answer', mode: 'followup', replyTo: 'msg-0' })
    expect(h.sink.delivered[0]?.replyTo).toBe('msg-0')
  })
})

/** One row of a `peer_list` result, as the model reads it. */
interface Row {
  readonly name: string
  readonly session_id: string
  readonly state: string
  readonly is_self: boolean
  readonly cwd?: string
  readonly task?: string
  readonly role?: string
  readonly working_on?: readonly string[]
}

interface Facts {
  readonly claims?: readonly { sessionId: string; resource: string; intent: string }[]
  readonly cards?: readonly CapabilityCard[]
  readonly states?: readonly TaskState[]
}

/** The tool as the agent loop registers it, over the same peers the send path sees. */
function peerListTool(peers: readonly PeerDescriptor[], facts: Facts = {}) {
  return createPeerListTool(
    harness(peers).sender,
    { all: () => Promise.resolve(facts.claims ?? []) } as unknown as WorkClaims,
    { readAll: () => Promise.resolve(facts.cards ?? []) } as unknown as CardStore,
    { readAll: () => Promise.resolve(facts.states ?? []) } as unknown as TaskStateStore,
    createIdentityResolver(new FakeDirectory(peers)),
  )
}

/** Run a tool on behalf of one session, the way the agent loop supplies its execution. */
function runAs(sessionId: string): ToolRunContext {
  return { agent: { id: sessionId }, signal: new AbortController().signal } as unknown as ToolRunContext
}

/** Call `peer_list` as one session: its rows, and the text a model reads. */
async function listAs(peers: readonly PeerDescriptor[], sessionId: string, facts: Facts = {}) {
  const tool = peerListTool(peers, facts)
  const rows = (await tool.execute({}, runAs(sessionId))) as Row[]
  const text = (tool.output.render({}, rows as never)[0] as { text: string }).text
  return { rows, text }
}

describe('peer_list: every session sees its own canonical ID', () => {
  // The worked example from the plugin's spec: one API worker and one frontend worker.
  const apiSession = peer({ sessionId: 'session-a', name: 'api-worker', cwd: '/project/api' })
  const frontendSession = peer({ sessionId: 'session-b', name: 'frontend-worker', cwd: '/project/frontend' })

  it('lists the caller first, marked is_self, and every other session marked is_self false', async () => {
    const { rows } = await listAs([apiSession, frontendSession], 'session-a')

    expect(rows.map((row) => row.session_id)).toEqual(['session-a', 'session-b'])
    expect(rows.map((row) => row.is_self)).toEqual([true, false])
    expect(rows[0]?.state).toBe('running')
    expect(rows[1]?.state).toBe('idle')
  })

  it('renders the full session ID of the caller and of each peer', async () => {
    const { text } = await listAs([apiSession, frontendSession], 'session-a')

    expect(text).toBe(
      [
        'SELF: api-worker [running]',
        '  session_id: session-a',
        '  cwd: /project/api',
        '',
        'PEER: frontend-worker [idle]',
        '  session_id: session-b',
        '  cwd: /project/frontend',
      ].join('\n'),
    )
  })

  it('shows the same canonical IDs from the other session, with the roles swapped', async () => {
    const { rows, text } = await listAs([apiSession, frontendSession], 'session-b')

    expect(rows.map((row) => row.session_id)).toEqual(['session-b', 'session-a'])
    expect(rows.map((row) => row.is_self)).toEqual([true, false])
    expect(text).toContain('SELF: frontend-worker [running]\n  session_id: session-b')
    expect(text).toContain('PEER: api-worker [idle]\n  session_id: session-a')
  })

  it('takes the caller from the execution, never from a model-supplied argument', async () => {
    const tool = peerListTool([apiSession, frontendSession])
    const rows = (await tool.execute({ session_id: 'session-b', is_self: true }, runAs('session-a'))) as Row[]

    expect(rows[0]?.session_id).toBe('session-a')
    expect(rows[0]?.is_self).toBe(true)
  })

  it('refuses to run without an owning session', async () => {
    const tool = peerListTool([apiSession, frontendSession])
    await expect(tool.execute({}, {} as ToolRunContext)).rejects.toThrow(/agent/)
  })

  it('lists the caller alone, and says so, when no other session is reachable', async () => {
    const { rows, text } = await listAs([apiSession], 'session-a')

    expect(rows).toHaveLength(1)
    expect(rows[0]?.is_self).toBe(true)
    expect(text).toBe(
      ['SELF: api-worker [running]', '  session_id: session-a', '  cwd: /project/api', '', 'No other sessions are reachable.'].join('\n'),
    )
  })

  it('lists the caller even when the directory does not yet know it', async () => {
    const { rows } = await listAs([frontendSession], 'session-z')

    expect(rows[0]).toMatchObject({ name: 'session-z', session_id: 'session-z', is_self: true })
    expect(rows[1]?.is_self).toBe(false)
  })

  it('names the caller by its published alias when it has one', async () => {
    const aliased = peer({ sessionId: 'session-a', name: 'api-worker', alias: 'api', cwd: '/project/api' })
    const { text } = await listAs([aliased, frontendSession], 'session-a')

    expect(text).toContain('SELF: api [running]')
  })

  it('shows a peer by its alias, and that alias, the exact ID and the name all route to it', async () => {
    const aliasedPeer = peer({ sessionId: 'session-f', name: 'frontend-worker-7f3a', alias: 'frontend', cwd: '/project/frontend' })
    const { text } = await listAs([apiSession, aliasedPeer], 'session-a')
    expect(text).toContain('PEER: frontend [idle]\n  session_id: session-f')

    const h = harness([apiSession, aliasedPeer])
    const byAlias = await h.sender.send(alice, { to: 'frontend', body: 'by alias', mode: 'followup' })
    const byId = await h.sender.send(alice, { to: 'session-f', body: 'by id', mode: 'followup' })
    const byName = await h.sender.send(alice, { to: 'frontend-worker-7f3a', body: 'by name', mode: 'followup' })
    expect([byAlias.peer.sessionId, byId.peer.sessionId, byName.peer.sessionId]).toEqual([
      'session-f',
      'session-f',
      'session-f',
    ])
  })

  it('still refuses to send to the caller, whether by its ID or by its alias', async () => {
    const aliasedSelf = peer({ sessionId: 'session-a', name: 'api-worker', alias: 'api' })
    const h = harness([aliasedSelf, frontendSession])

    await expect(h.sender.send(alice, { to: 'session-a', body: 'hi', mode: 'followup' })).rejects.toThrow(PeerError)
    await expect(h.sender.send(alice, { to: 'api', body: 'hi', mode: 'followup' })).rejects.toThrow(PeerError)
  })

  it('lists sessions in other host processes with their full IDs and where they run', async () => {
    const remote = peer({
      sessionId: 'session-r',
      name: 'remote-worker',
      location: { kind: 'remote', hostId: 'h2', socketPath: '/tmp/h2.sock' },
    })
    const { rows, text } = await listAs([apiSession, frontendSession, remote], 'session-a')

    expect(rows.map((row) => row.session_id)).toEqual(['session-a', 'session-b', 'session-r'])
    expect(text).toContain('PEER: remote-worker [running (another host process)]\n  session_id: session-r')
  })

  it('keeps each row role, task and claims, including the caller own', async () => {
    const facts: Facts = {
      cards: [createCard({ sessionId: 'session-b', role: 'frontend UI', now: 0 })],
      states: [
        createTaskState({
          sessionId: 'session-b',
          name: 'frontend-worker',
          phase: 'blocked',
          summary: 'waiting for the schema',
          blockedOn: 'session-a',
          now: 0,
        }),
      ],
      claims: [{ sessionId: 'session-a', resource: 'src/api.ts', intent: 'editing' }],
    }
    const { rows, text } = await listAs([apiSession, frontendSession], 'session-a', facts)

    const frontendRow = rows.find((row) => row.session_id === 'session-b')
    expect(frontendRow?.role).toContain('frontend UI')
    // The wait names the caller by the same address its own row shows.
    expect(frontendRow?.task).toBe('blocked on api-worker: waiting for the schema')
    expect(rows[0]?.working_on).toEqual(['src/api.ts (editing)'])
    expect(text).toContain('  working on: src/api.ts (editing)')
  })
})
