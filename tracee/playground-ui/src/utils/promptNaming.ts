import axios from 'axios';

export function slugifyPromptName(name: string) {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-')
    .slice(0, 48);

  return slug || 'playground-prompt';
}

async function promptExists(
  promptId: string,
  getPrompt: (promptId: string) => Promise<unknown>,
) {
  return getPrompt(promptId).then(
    () => true,
    (error) => {
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        return false;
      }

      throw error;
    },
  );
}

export async function generateUniquePromptId(
  promptName: string,
  getPrompt: (promptId: string) => Promise<unknown>,
) {
  const baseId = slugifyPromptName(promptName);

  for (let index = 0; index < 25; index += 1) {
    const candidateId = index === 0 ? baseId : `${baseId}-${index + 1}`;
    const exists = await promptExists(candidateId, getPrompt);
    if (!exists) {
      return candidateId;
    }
  }

  return `${baseId}-${Date.now().toString(36)}`;
}
