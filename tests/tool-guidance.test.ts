import { describe, expect, it } from 'vitest'

import { createPeerListTool } from '../src/tools/peer-list.ts'
import { createPeerSendTool } from '../src/tools/peer-send.ts'
import type { MessageSender } from '../src/app/send-message.ts'
import type { WorkClaims } from '../src/app/claim-work.ts'
import type { CardStore } from '../src/adapters/cards.ts'
import type { TaskStateStore } from '../src/adapters/task-states.ts'
import { SKILL_BODY } from '../src/skill.ts'

/**
 * Guidance that a live run proved load-bearing.
 *
 * These are one-sentence rules in a tool description, which is exactly the kind
 * of thing an edit removes without anyone noticing — and each one is here
 * because its absence produced a real failure in a real session.
 */
const sendTool = createPeerSendTool({} as MessageSender, async () => ({
  sessionId: 's',
  name: 'n',
}))

describe('peer_send guidance', () => {
  it('forbids acknowledgement traffic', () => {
    // Two sessions in a live run wound down an exchange with "Noted, thanks",
    // "Anytime", "Perfect — I'm here" — four turns spent on politeness, ended
    // only by the loop guard dropping a duplicate. Every message costs the
    // receiver a turn, and courtesy is the cheapest way to waste one.
    expect(sendTool.description).toMatch(/acknowledgements/i)
    expect(sendTool.description).toMatch(/costs the receiver a turn/i)
    expect(sendTool.description).toMatch(/stop replying/i)
  })

  it('keeps the sender from treating a reply as permission', () => {
    expect(sendTool.description).toMatch(/do not treat\s+its reply as permission/i)
  })

  it('still explains what each delivery mode costs', () => {
    expect(sendTool.description).toMatch(/steer.*interrupts/i)
    expect(sendTool.description).toMatch(/context.*without waking it/i)
  })
})

const listTool = createPeerListTool(
  {} as MessageSender,
  {} as WorkClaims,
  {} as CardStore,
  {} as TaskStateStore,
  async () => ({ sessionId: 's', name: 'n' }),
)

describe('peer_list guidance', () => {
  it('marks the calling session SELF and says it cannot be messaged', () => {
    expect(listTool.description).toMatch(/first entry, marked SELF/)
    expect(listTool.description).toMatch(/cannot send to SELF/)
  })

  it('points a model at the full session_id when it needs an exact ID', () => {
    expect(listTool.description).toMatch(/full session_id/)
  })

  it('tells the coordination skill when to call peer_list for the session-ID mapping', () => {
    expect(SKILL_BODY).toContain('`peer_list` when you need the current session-ID mapping')
    expect(SKILL_BODY).toContain('SELF')
  })
})
