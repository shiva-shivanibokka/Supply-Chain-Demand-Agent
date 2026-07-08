export type HistoryPoint = { date: string; demand: number };

export type PartRecord = {
  part_id: string;
  category: string;
  supplier: string;
  region: string;
  lead_time_days: number;
  price_usd: number;
  inventory: number;
  avg_daily_demand: number; // mean demand over last 30 days
  history: HistoryPoint[]; // last 90 days, chronological
};
