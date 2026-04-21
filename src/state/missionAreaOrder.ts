export function appendMissionAreaOrderIds(order: string[], ids: string[]): string[] {
  if (ids.length === 0) return order;
  const seen = new Set(order);
  const appended: string[] = [];
  for (const id of ids) {
    if (!id || seen.has(id)) continue;
    seen.add(id);
    appended.push(id);
  }
  return appended.length > 0 ? [...order, ...appended] : order;
}

export function removeMissionAreaOrderIds(order: string[], ids: string[]): string[] {
  if (ids.length === 0) return order;
  const toRemove = new Set(ids);
  const next = order.filter((id) => !toRemove.has(id));
  return next.length === order.length ? order : next;
}

export function replaceMissionAreaOrderIds(order: string[], affectedIds: string[], replacementIds: string[]): string[] {
  if (affectedIds.length === 0) return appendMissionAreaOrderIds(order, replacementIds);
  const affected = new Set(affectedIds);
  const replacement = replacementIds.filter((id) => id.length > 0);
  const next: string[] = [];
  let inserted = false;
  for (const id of order) {
    if (!affected.has(id)) {
      next.push(id);
      continue;
    }
    if (!inserted) {
      next.push(...replacement);
      inserted = true;
    }
  }
  if (!inserted) {
    next.push(...replacement);
  }
  return appendMissionAreaOrderIds([], next);
}
