"use client";

import { PROVIDERS, type ProviderName } from "@/lib/providers";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";

export function ProviderBar({
  provider,
  onProviderChange,
  model,
  onModelChange,
  apiKey,
  onApiKeyChange,
}: {
  provider: ProviderName;
  onProviderChange: (provider: ProviderName) => void;
  model: string;
  onModelChange: (model: string) => void;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
}) {
  const config = PROVIDERS[provider];

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-card/40 p-3">
      <Select
        value={provider}
        onValueChange={(value) => {
          const next = value as ProviderName;
          onProviderChange(next);
          onModelChange(PROVIDERS[next].models[0]);
        }}
      >
        <SelectTrigger className="w-44">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(PROVIDERS) as ProviderName[]).map((name) => (
            <SelectItem key={name} value={name}>
              {PROVIDERS[name].label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Select value={model} onValueChange={(value) => onModelChange(value as string)}>
        <SelectTrigger className="w-56">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {config.models.map((m) => (
            <SelectItem key={m} value={m}>
              {m}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Input
        type="password"
        placeholder={config.keyPlaceholder}
        value={apiKey}
        onChange={(e) => onApiKeyChange(e.target.value)}
        autoComplete="off"
        className="max-w-56"
      />

      <span className="text-xs text-muted-foreground">
        {config.free ? "Free tier available" : "Bring your own key"} — key stays in this
        browser tab only, never saved.
      </span>
    </div>
  );
}
