import { appendBounded, streamCandidates } from '../api';

test('status log keeps only the most recent 30 events', () => {
  const events = Array.from({ length: 35 }, (_, attempt) => ({ attempt }));
  const bounded = events.reduce((items, event) => appendBounded(items, event, 30), [] as typeof events);
  expect(bounded).toHaveLength(30);
  expect(bounded[0].attempt).toBe(5);
  expect(bounded[29].attempt).toBe(34);
});

test('candidate SSE parser dispatches progress and completion across chunks', async () => {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode('event: progress\ndata: {"phase":"SAMPLING","attempt":1,"accepted":0,"invalid":0,"duplicates":0,"rejected":0,"reviews":0,"found":0,"unverified":0,"detail":null}\n'));
      controller.enqueue(encoder.encode('\nevent: complete\ndata: {"shortlist":[],"review_queue":[],"summary":{"attempts":1}}\n\n'));
      controller.close();
    }
  });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(stream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })));
  const progress = vi.fn();
  const complete = vi.fn();
  await streamCandidates(
    { target_descriptors: ['floral'], sampling_diversity: 0.8, reference_consents: ['PUBCHEM'] },
    { onProgress: progress, onComplete: complete, onError: vi.fn() },
    new AbortController().signal
  );
  expect(progress).toHaveBeenCalledWith(expect.objectContaining({ phase: 'SAMPLING', attempt: 1 }));
  expect(complete).toHaveBeenCalledWith(expect.objectContaining({ shortlist: [] }));
});
