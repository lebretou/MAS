export function resizeTextarea(textarea: HTMLTextAreaElement | null) {
  if (!textarea) {
    return;
  }

  textarea.style.height = '0px';

  const computedMaxHeight = Number.parseFloat(globalThis.getComputedStyle(textarea).maxHeight);
  const maxHeight = Number.isFinite(computedMaxHeight) ? computedMaxHeight : textarea.scrollHeight;
  const nextHeight = Math.min(textarea.scrollHeight, maxHeight);

  textarea.style.height = `${nextHeight}px`;
  textarea.style.overflowY = textarea.scrollHeight > nextHeight ? 'auto' : 'hidden';
}
