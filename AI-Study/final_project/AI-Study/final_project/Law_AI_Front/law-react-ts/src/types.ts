export type Role = 'user' | 'assistant';

export interface Message {
  role: Role;
  text: string;
}

export type PageKey = 'research' | 'simulation' | 'evidence' | 'history' | 'settings';

export interface AttachmentsState {
  images: File[];
  audio: File | null;
  files: File[];
}
