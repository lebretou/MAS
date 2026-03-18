/**
 * Cosine similarity utilities for comparing JSON outputs.
 * Uses weighted hybrid scoring: 75% structural + 25% value similarity.
 */

/** Strip markdown code fences (```json ... ```) that LLMs often wrap around JSON. */
function stripMarkdownFences(s: string): string {
  return s.replace(/^```\w*\n?/, '').replace(/\n?```$/, '').trim();
}

/** Structural tokens: key paths and type markers (schema shape). */
function flattenStructural(obj: unknown, prefix: string = '', visited: Set<object> = new Set()): string[] {
  const tokens: string[] = [];
  if (obj === null || typeof obj !== 'object') {
    tokens.push(`${prefix}:leaf`);
    return tokens;
  }
  if (visited.has(obj)) return tokens;
  visited.add(obj);
  if (Array.isArray(obj)) {
    tokens.push(`${prefix}:array`);
    obj.forEach((item, i) => {
      tokens.push(...flattenStructural(item, `${prefix}[${i}]`, visited));
    });
    return tokens;
  }
  tokens.push(`${prefix}:object`);
  for (const key of Object.keys(obj as Record<string, unknown>)) {
    const path = prefix ? `${prefix}.${key}` : key;
    tokens.push(`key:${path}`);
    tokens.push(...flattenStructural((obj as Record<string, unknown>)[key], path, visited));
  }
  return tokens;
}

/** Value tokens: leaf values keyed by their path. */
function flattenValues(obj: unknown, prefix: string = '', visited: Set<object> = new Set()): string[] {
  const tokens: string[] = [];
  if (obj === null) {
    tokens.push(`${prefix}=null`);
    return tokens;
  }
  if (typeof obj !== 'object') {
    tokens.push(`${prefix}=${String(obj)}`);
    return tokens;
  }
  if (visited.has(obj)) return tokens;
  visited.add(obj);
  if (Array.isArray(obj)) {
    obj.forEach((item, i) => {
      tokens.push(...flattenValues(item, `${prefix}[${i}]`, visited));
    });
    return tokens;
  }
  for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
    const path = prefix ? `${prefix}.${key}` : key;
    tokens.push(...flattenValues(value, path, visited));
  }
  return tokens;
}

function buildVocabulary(tokenSets: string[][]): Map<string, number> {
  const vocab = new Map<string, number>();
  let index = 0;
  for (const tokens of tokenSets) {
    for (const token of tokens) {
      if (!vocab.has(token)) {
        vocab.set(token, index++);
      }
    }
  }
  return vocab;
}

function toVector(tokens: string[], vocab: Map<string, number>): number[] {
  const vec = new Array(vocab.size).fill(0);
  for (const token of tokens) {
    const idx = vocab.get(token);
    if (idx !== undefined) vec[idx] += 1;
  }
  return vec;
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

export interface SimilarityResult {
  matrix: number[][]; // NxN similarity matrix
  points2D: Array<{ x: number; y: number; index: number }>; // 2D projection
  averageSimilarity: number[]; // average similarity per run
}

/** Sentinel for empty/null outputs — treated as failures. */
const EMPTY_SENTINEL = ['__empty__'];

interface TokenizedOutput {
  structural: string[];
  values: string[];
  isFailed: boolean;
}

function tokenizeOutput(output: string): TokenizedOutput {
  // Empty/null → failure
  if (!output || output.trim() === '' || output.trim() === 'null' || output.trim() === 'undefined') {
    return { structural: EMPTY_SENTINEL, values: EMPTY_SENTINEL, isFailed: true };
  }

  // Strip markdown fences before parsing
  const stripped = stripMarkdownFences(output);

  try {
    const parsed = JSON.parse(stripped);
    return {
      structural: flattenStructural(parsed),
      values: flattenValues(parsed),
      isFailed: false,
    };
  } catch {
    // Non-JSON: use word tokens for both (no structural/value split possible)
    const words = stripped.toLowerCase().split(/\s+/).filter(Boolean);
    if (words.length === 0) {
      return { structural: EMPTY_SENTINEL, values: EMPTY_SENTINEL, isFailed: true };
    }
    return { structural: words, values: words, isFailed: false };
  }
}

/**
 * Compute weighted hybrid cosine similarity (75% structural + 25% value) and project to 2D.
 * Accepts raw strings (will attempt JSON parse) or pre-parsed objects.
 */
export function computeSimilarity(outputs: string[]): SimilarityResult {
  const tokenized = outputs.map(tokenizeOutput);

  // Build separate vocabularies for structural and value tokens
  const structVocab = buildVocabulary(tokenized.map(t => t.structural));
  const valueVocab = buildVocabulary(tokenized.map(t => t.values));

  const structVectors = tokenized.map(t => toVector(t.structural, structVocab));
  const valueVectors = tokenized.map(t => toVector(t.values, valueVocab));

  // Compute NxN weighted hybrid similarity matrix
  const n = outputs.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = 1;
      } else if (tokenized[i].isFailed || tokenized[j].isFailed) {
        matrix[i][j] = 0;
      } else {
        const structSim = cosine(structVectors[i], structVectors[j]);
        const valueSim = cosine(valueVectors[i], valueVectors[j]);
        matrix[i][j] = 0.75 * structSim + 0.25 * valueSim;
      }
    }
  }

  // Average similarity per run
  const averageSimilarity = matrix.map((row, i) => {
    const others = row.filter((_, j) => j !== i);
    return others.length > 0 ? others.reduce((a, b) => a + b, 0) / others.length : 1;
  });

  // MDS projection with centroid-attraction radial layout
  const points2D = project2D(matrix, averageSimilarity);

  return { matrix, points2D, averageSimilarity };
}

/**
 * Classical MDS (Multidimensional Scaling) projection to 2D.
 * Uses double-centering of the squared distance matrix to extract
 * the two principal coordinates. Similar runs cluster together;
 * outliers appear far from the group.
 */
function project2D(similarity: number[][], averageSimilarity: number[]): Array<{ x: number; y: number; index: number }> {
  const n = similarity.length;
  if (n <= 1) return [{ x: 0, y: 0, index: 0 }];
  if (n === 2) {
    const r0 = Math.pow(1 - averageSimilarity[0], 2);
    const r1 = Math.pow(1 - averageSimilarity[1], 2);
    return [
      { x: 0.85 - r0 * 0.3, y: 0.5, index: 0 },
      { x: 0.85 + r1 * 0.3, y: 0.5, index: 1 },
    ];
  }

  // Step 1: Squared distance matrix D² with cubic amplification.
  // D²[i][j] = (1 - similarity)^3 so similar runs collapse together
  // while outliers get pushed far apart.
  // sim 0.95 → 0.000125, sim 0.7 → 0.027, sim 0.3 → 0.343
  const D2: number[][] = similarity.map(row => row.map(s => {
    return Math.pow(Math.max(0, 1 - s), 3);
  }));

  // Step 2: Double-centering → B = -0.5 * H * D² * H  where H = I - (1/n)*11'
  // Row means, column means, grand mean
  const rowMeans = D2.map(row => row.reduce((a, b) => a + b, 0) / n);
  const colMeans: number[] = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) colMeans[j] += D2[i][j];
    colMeans[j] /= n;
  }
  let grandMean = 0;
  for (let i = 0; i < n; i++) grandMean += rowMeans[i];
  grandMean /= n;

  const B: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      -0.5 * (D2[i][j] - rowMeans[i] - colMeans[j] + grandMean)
    )
  );

  // Step 3: Jacobi eigendecomposition to find top 2 eigenvectors of B
  const { values, vectors } = jacobiEigen(B, n);

  // Pick the two largest positive eigenvalues
  const indexed = values
    .map((v, i) => ({ value: v, index: i }))
    .filter(e => e.value > 1e-10)
    .sort((a, b) => b.value - a.value);

  const i1 = indexed.length > 0 ? indexed[0].index : 0;
  const s1 = indexed.length > 0 ? Math.sqrt(indexed[0].value) : 0;

  // Degenerate case: only one positive eigenvalue (all runs nearly identical).
  // Use circular fallback for y-axis to avoid collapsing all points to a line.
  const hasTwoDimensions = indexed.length > 1;
  const i2 = hasTwoDimensions ? indexed[1].index : 0;
  const s2 = hasTwoDimensions ? Math.sqrt(indexed[1].value) : 0;

  // Step 4: Project — coordinates = eigenvector * sqrt(eigenvalue)
  const GOLDEN_ANGLE = 2.399;
  const raw = Array.from({ length: n }, (_, i) => ({
    x: vectors[i][i1] * s1,
    y: hasTwoDimensions
      ? vectors[i][i2] * s2
      : Math.sin(i * GOLDEN_ANGLE) * 0.05, // circular jitter fallback
    index: i,
  }));

  // Step 5: Centroid-attraction radial layout
  // Compute centroid of raw MDS points
  let cx = 0, cy = 0;
  for (const p of raw) {
    cx += p.x;
    cy += p.y;
  }
  cx /= n;
  cy /= n;

  // Chart-space centroid: cluster on the far right at (0.85, 0.5)
  const chartCx = 0.85;
  const chartCy = 0.5;

  const result: Array<{ x: number; y: number; index: number }> = [];

  for (let i = 0; i < n; i++) {
    const sim = averageSimilarity[i];
    const isFailed = sim === 0;

    // Radius: (1 - sim)² — high similarity → tiny radius, low → large
    const radius = isFailed ? 1.0 : Math.pow(1 - sim, 2);

    // Direction from MDS centroid (preserves angular relationships)
    let dx = raw[i].x - cx;
    let dy = raw[i].y - cy;
    const mag = Math.sqrt(dx * dx + dy * dy);

    if (isFailed) {
      // Failed runs point left
      dx = -1;
      dy = 0;
    } else if (mag > 1e-10) {
      // Normalize to unit vector
      dx /= mag;
      dy /= mag;
    } else {
      // Overlapping with centroid — use spiral jitter angle
      const angle = (i / n) * 2 * Math.PI;
      dx = Math.cos(angle);
      dy = Math.sin(angle);
    }

    // Small spiral jitter to prevent exact overlaps
    const jitterAngle = i * GOLDEN_ANGLE;
    const jitterMag = i * 0.008;
    const jx = Math.cos(jitterAngle) * jitterMag;
    const jy = Math.sin(jitterAngle) * jitterMag;

    result.push({
      x: chartCx + dx * radius + jx,
      y: chartCy + dy * radius + jy,
      index: raw[i].index,
    });
  }

  // Clamp to [0, 1] — no rescaling (preserves centroid-attraction layout)
  for (const p of result) {
    p.x = Math.max(0, Math.min(1, p.x));
    p.y = Math.max(0, Math.min(1, p.y));
  }

  return result;
}

/**
 * Jacobi eigendecomposition for a symmetric matrix.
 * Reliable for small matrices (n <= 20). Returns all eigenvalues and eigenvectors.
 */
function jacobiEigen(
  M: number[][],
  n: number,
  maxIter: number = 200
): { values: number[]; vectors: number[][] } {
  // Work on a copy
  const A = M.map(row => [...row]);
  // Initialize eigenvector matrix as identity
  const V: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );

  for (let iter = 0; iter < maxIter; iter++) {
    // Find largest off-diagonal element
    let maxVal = 0, p = 0, q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(A[i][j]) > maxVal) {
          maxVal = Math.abs(A[i][j]);
          p = i;
          q = j;
        }
      }
    }

    // Convergence check
    if (maxVal < 1e-12) break;

    // Compute rotation angle
    const theta = A[p][p] === A[q][q]
      ? Math.PI / 4
      : 0.5 * Math.atan2(2 * A[p][q], A[p][p] - A[q][q]);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    // Apply Givens rotation to A
    const App = A[p][p], Aqq = A[q][q], Apq = A[p][q];
    A[p][p] = c * c * App + 2 * s * c * Apq + s * s * Aqq;
    A[q][q] = s * s * App - 2 * s * c * Apq + c * c * Aqq;
    A[p][q] = 0;
    A[q][p] = 0;

    for (let i = 0; i < n; i++) {
      if (i === p || i === q) continue;
      const Aip = A[i][p], Aiq = A[i][q];
      A[i][p] = c * Aip + s * Aiq;
      A[p][i] = A[i][p];
      A[i][q] = -s * Aip + c * Aiq;
      A[q][i] = A[i][q];
    }

    // Accumulate eigenvectors
    for (let i = 0; i < n; i++) {
      const Vip = V[i][p], Viq = V[i][q];
      V[i][p] = c * Vip + s * Viq;
      V[i][q] = -s * Vip + c * Viq;
    }
  }

  const values = Array.from({ length: n }, (_, i) => A[i][i]);
  return { values, vectors: V };
}
