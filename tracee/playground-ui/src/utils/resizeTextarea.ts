export function resizeTextarea(textarea: HTMLTextAreaElement | null) {
  if (!textarea) {
    return;
  }

  textarea.style.height = '0px';
  textarea.style.height = `${textarea.scrollHeight}px`;
  textarea.style.overflowY = 'hidden';
}
