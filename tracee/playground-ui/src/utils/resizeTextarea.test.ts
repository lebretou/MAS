import { describe, expect, it } from 'vitest';
import { resizeTextarea } from './resizeTextarea';

function createTextarea(scrollHeight: number) {
  return {
    scrollHeight,
    style: {
      height: '',
      overflowY: '',
    },
  } as unknown as HTMLTextAreaElement;
}

describe('resizeTextarea', () => {
  it('expands to fit the full textarea content', () => {
    const textarea = createTextarea(248);

    resizeTextarea(textarea);

    expect(textarea.style.height).toBe('248px');
    expect(textarea.style.overflowY).toBe('hidden');
  });

  it('ignores null textareas', () => {
    expect(() => resizeTextarea(null)).not.toThrow();
  });
});
