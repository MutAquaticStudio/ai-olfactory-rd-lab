import type {
  AnalysisResult,
  AssessmentPayload,
  AppMeta,
  DatasetVersion,
  GenerationComplete,
  GenerationEvent,
  ProductError,
  ImportValidation
} from './types';

export class ApiError extends Error {
  code: string;
  technicalDetails?: string;

  constructor(error: ProductError) {
    super(error.message);
    this.name = 'ApiError';
    this.code = error.code;
    this.technicalDetails = error.technical_details;
  }
}

export function appendBounded<T>(items: T[], next: T, limit: number): T[] {
  return [...items, next].slice(-Math.max(0, limit));
}

async function parseError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    const detail = body.detail ?? body;
    return new ApiError({
      code: detail.code ?? `HTTP_${response.status}`,
      message: detail.message ?? 'The request could not be completed.',
      technical_details: detail.technical_details
    });
  } catch {
    return new ApiError({ code: `HTTP_${response.status}`, message: 'The request could not be completed.' });
  }
}

export async function getMeta(signal?: AbortSignal): Promise<AppMeta> {
  const response = await fetch('/api/v1/meta', { signal });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function analyzeMolecule(smiles: string, signal?: AbortSignal): Promise<AnalysisResult> {
  const response = await fetch('/api/v1/analysis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smiles }),
    signal
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function validateAssessment(
  payload: AssessmentPayload,
  signal?: AbortSignal
): Promise<ImportValidation> {
  const response = await fetch('/api/v1/assessments/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function commitAssessment(payload: AssessmentPayload, signal?: AbortSignal) {
  const response = await fetch('/api/v1/assessments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function validateImport(file: File, signal?: AbortSignal): Promise<ImportValidation> {
  const form = new FormData();
  form.append('file', file);
  const response = await fetch('/api/v1/data/imports/validate', {
    method: 'POST',
    body: form,
    signal
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function commitImport(validationToken: string, signal?: AbortSignal) {
  const response = await fetch('/api/v1/data/imports/commit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ validation_token: validationToken }),
    signal
  });
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function getDatasetVersions(signal?: AbortSignal): Promise<DatasetVersion[]> {
  const response = await fetch('/api/v1/datasets/versions', { signal });
  if (!response.ok) throw await parseError(response);
  const result = await response.json();
  return result.versions;
}

interface StreamCallbacks {
  onProgress: (event: GenerationEvent) => void;
  onComplete: (result: GenerationComplete) => void;
  onError: (error: ProductError) => void;
}

function dispatchBlock(block: string, callbacks: StreamCallbacks) {
  let eventName = 'message';
  const data: string[] = [];
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) eventName = line.slice(6).trim();
    if (line.startsWith('data:')) data.push(line.slice(5).trim());
  }
  if (!data.length) return;
  const payload = JSON.parse(data.join('\n'));
  if (eventName === 'progress') callbacks.onProgress(payload as GenerationEvent);
  if (eventName === 'complete') callbacks.onComplete(payload as GenerationComplete);
  if (eventName === 'error') callbacks.onError(payload as ProductError);
}

export async function streamCandidates(
  request: {
    target_descriptors: string[];
    sampling_diversity: number;
    reference_consents: string[];
    pubchem_consent?: boolean;
  },
  callbacks: StreamCallbacks,
  signal: AbortSignal
) {
  const response = await fetch('/api/v1/candidates/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(request),
    signal
  });
  if (!response.ok) throw await parseError(response);
  if (!response.body) throw new ApiError({ code: 'STREAM_UNAVAILABLE', message: 'Live status is unavailable.' });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, '\n');
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop() ?? '';
    blocks.filter(Boolean).forEach((block) => dispatchBlock(block, callbacks));
    if (done) break;
  }
  if (buffer.trim()) dispatchBlock(buffer, callbacks);
}
