export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

export function describeEvent(event: unknown): {
  title: string;
  detail: string | null;
  url: string | null;
} {
  if (typeof event === "string") {
    return { title: event, detail: null, url: null };
  }

  if (event && typeof event === "object") {
    const record = event as Record<string, unknown>;
    const titleValue = record.title ?? record.name ?? record.type ?? record.source;
    const detailValue = record.detail ?? record.description ?? record.query ?? record.content;
    return {
      title: typeof titleValue === "string" ? titleValue : "Activity",
      detail:
        typeof detailValue === "string"
          ? detailValue
          : Object.keys(record).length
            ? JSON.stringify(record, null, 2)
            : null,
      url: typeof record.url === "string" ? record.url : null,
    };
  }

  return { title: String(event), detail: null, url: null };
}
