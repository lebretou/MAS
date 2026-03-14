export type DeviationType = 'missing' | 'type_mismatch' | 'extra';

export interface FieldSchema {
  path: string;
  type: string; // 'string' | 'number' | 'boolean' | 'null' | 'object' | 'array'
  count: number; // how many runs have this field
}

export interface ConsensusSchema {
  fields: FieldSchema[];
  totalRuns: number;
}

export interface FieldDeviation {
  path: string;
  type: DeviationType;
  expected?: string;
  actual?: string;
}

export type RunClassification = 'conforming' | 'minor_deviation' | 'major_deviation' | 'failure';

const MAJOR_DEVIATION_THRESHOLD = 3;

function getType(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function flattenKeys(
  obj: unknown,
  prefix: string = '',
  visited: Set<object> = new Set(),
): Array<{ path: string; type: string }> {
  const results: Array<{ path: string; type: string }> = [];
  if (obj === null || typeof obj !== 'object') return results;
  if (visited.has(obj)) return results;
  visited.add(obj);

  const entries = Array.isArray(obj)
    ? obj.map((v, i) => [`[${i}]`, v] as const)
    : Object.entries(obj as Record<string, unknown>);

  for (const [key, value] of entries) {
    const fullPath = prefix ? `${prefix}.${key}` : key;
    results.push({ path: fullPath, type: getType(value) });
    if (value !== null && typeof value === 'object') {
      results.push(...flattenKeys(value, fullPath, visited));
    }
  }
  return results;
}

export function inferSchema(jsonOutput: unknown): Array<{ path: string; type: string }> {
  if (jsonOutput === null || typeof jsonOutput !== 'object') return [];
  return flattenKeys(jsonOutput);
}

export function computeConsensusSchema(
  parsedOutputs: Array<{ parsed: unknown; index: number }>
): ConsensusSchema {
  const fieldMap = new Map<string, FieldSchema>();
  const typeFrequency = new Map<string, Map<string, number>>();
  const totalRuns = parsedOutputs.length;

  for (const { parsed } of parsedOutputs) {
    const fields = inferSchema(parsed);
    const seenPaths = new Set<string>();

    for (const { path, type } of fields) {
      if (seenPaths.has(path)) continue;
      seenPaths.add(path);

      const existing = fieldMap.get(path);
      if (existing) {
        fieldMap.set(path, { ...existing, count: existing.count + 1 });
      } else {
        fieldMap.set(path, { path, type, count: 1 });
      }

      // Track type frequency for majority vote
      const pathFreq = typeFrequency.get(path) ?? new Map<string, number>();
      pathFreq.set(type, (pathFreq.get(type) ?? 0) + 1);
      typeFrequency.set(path, pathFreq);
    }
  }

  // Resolve majority type for each field
  const fields = Array.from(fieldMap.values()).map(field => {
    const freqs = typeFrequency.get(field.path);
    if (!freqs) return field;

    let majorityType = field.type;
    let maxCount = 0;
    freqs.forEach((count, type) => {
      if (count > maxCount) {
        maxCount = count;
        majorityType = type;
      }
    });
    return { ...field, type: majorityType };
  });

  return { fields, totalRuns };
}

export function findDeviations(
  parsed: unknown,
  consensus: ConsensusSchema
): FieldDeviation[] {
  const deviations: FieldDeviation[] = [];
  const runFields = inferSchema(parsed);
  const runFieldMap = new Map(runFields.map(f => [f.path, f.type]));
  const consensusFieldMap = new Map(consensus.fields.map(f => [f.path, f]));

  const majorityThreshold = consensus.totalRuns / 2;

  // Check for missing fields (present in majority but not in this run)
  for (const field of consensus.fields) {
    if (field.count >= majorityThreshold && !runFieldMap.has(field.path)) {
      deviations.push({
        path: field.path,
        type: 'missing',
        expected: field.type,
      });
    }
  }

  // Check for type mismatches and extra fields
  for (const { path, type } of runFields) {
    const consensusField = consensusFieldMap.get(path);
    if (!consensusField) {
      deviations.push({ path, type: 'extra', actual: type });
    } else if (consensusField.type !== type && consensusField.count >= majorityThreshold) {
      deviations.push({
        path,
        type: 'type_mismatch',
        expected: consensusField.type,
        actual: type,
      });
    }
  }

  return deviations;
}

export function classifyRun(deviations: FieldDeviation[]): RunClassification {
  if (deviations.length === 0) return 'conforming';
  const missingCount = deviations.filter(d => d.type === 'missing').length;
  const typeCount = deviations.filter(d => d.type === 'type_mismatch').length;
  const totalSerious = missingCount + typeCount;
  if (totalSerious >= MAJOR_DEVIATION_THRESHOLD) return 'major_deviation';
  if (totalSerious >= 1) return 'minor_deviation';
  return 'conforming'; // only extras
}

export function getClassificationColor(classification: RunClassification): string {
  switch (classification) {
    case 'conforming': return '#065f46';
    case 'minor_deviation': return '#92400e';
    case 'major_deviation': return '#991b1b';
    case 'failure': return '#991b1b';
  }
}

export function getClassificationBg(classification: RunClassification): string {
  switch (classification) {
    case 'conforming': return '#ecfdf5';
    case 'minor_deviation': return '#fffbeb';
    case 'major_deviation': return '#fef2f2';
    case 'failure': return '#fef2f2';
  }
}
