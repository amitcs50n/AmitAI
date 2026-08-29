import type { Conversation } from "@/lib/types";

export type ConversationGroup = "Today" | "Yesterday" | "Previous 7 days" | "Earlier";

export const CONVERSATION_GROUPS: ConversationGroup[] = [
  "Today",
  "Yesterday",
  "Previous 7 days",
  "Earlier",
];

export function groupConversations(
  conversations: Conversation[],
  now = new Date(),
): Record<ConversationGroup, Conversation[]> {
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startYesterday = new Date(startToday);
  startYesterday.setDate(startYesterday.getDate() - 1);
  const startWeek = new Date(startToday);
  startWeek.setDate(startWeek.getDate() - 7);

  const groups: Record<ConversationGroup, Conversation[]> = {
    Today: [],
    Yesterday: [],
    "Previous 7 days": [],
    Earlier: [],
  };

  for (const conversation of conversations) {
    const updated = new Date(conversation.updated_at);
    if (updated >= startToday) {
      groups.Today.push(conversation);
    } else if (updated >= startYesterday) {
      groups.Yesterday.push(conversation);
    } else if (updated >= startWeek) {
      groups["Previous 7 days"].push(conversation);
    } else {
      groups.Earlier.push(conversation);
    }
  }

  return groups;
}

export function formatMessageTime(timestamp: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(timestamp));
}
