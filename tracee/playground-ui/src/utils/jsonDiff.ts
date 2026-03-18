/**
 * Simple LCS-based unified diff for comparing two text strings line-by-line.
 * No external dependencies.
 */

export interface DiffLine {
  type: 'same' | 'added' | 'removed';
  text: string;
}

/**
 * Compute the longest common subsequence table for two arrays of lines.
 */
function lcsTable(a: string[], b: string[]): number[][] {
  const m = a.length;
  const n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1] + 1
        : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }
  return dp;
}

/**
 * Produce a unified diff (list of same/added/removed lines) between two strings.
 * `base` is the consensus output, `target` is the selected run's output.
 */
const MAX_DIFF_LINES = 500;

export function computeDiff(base: string, target: string): DiffLine[] {
  const aLines = base.split('\n');
  const bLines = target.split('\n');

  if (aLines.length > MAX_DIFF_LINES || bLines.length > MAX_DIFF_LINES) {
    return [{ type: 'same', text: `[Diff truncated — outputs exceed ${MAX_DIFF_LINES} lines]` }];
  }

  const dp = lcsTable(aLines, bLines);

  const result: DiffLine[] = [];
  let i = aLines.length;
  let j = bLines.length;

  // Backtrack through LCS table
  const stack: DiffLine[] = [];
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && aLines[i - 1] === bLines[j - 1]) {
      stack.push({ type: 'same', text: aLines[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      stack.push({ type: 'added', text: bLines[j - 1] });
      j--;
    } else {
      stack.push({ type: 'removed', text: aLines[i - 1] });
      i--;
    }
  }

  // Reverse since we built it backwards
  for (let k = stack.length - 1; k >= 0; k--) {
    result.push(stack[k]);
  }

  return result;
}

/**
 * Find the majority consensus output: the run with the highest average similarity.
 * Returns the index into the outputs array, or -1 if none found.
 */
export function findConsensusOutputIndex(
  averageSimilarity: number[],
  validIndices: number[],
): number {
  if (validIndices.length === 0) return -1;

  let bestIdx = validIndices[0];
  let bestSim = -1;

  for (let i = 0; i < validIndices.length; i++) {
    const sim = averageSimilarity[i] ?? 0;
    if (sim > bestSim) {
      bestSim = sim;
      bestIdx = validIndices[i];
    }
  }

  return bestIdx;
}
