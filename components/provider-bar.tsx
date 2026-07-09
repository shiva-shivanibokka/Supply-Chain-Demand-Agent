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
import { Button } from "@/components/ui/button";

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
    <div className="panel-edge flex flex-col gap-2 rounded-xl border border-border bg-card/40 p-3">
      <div className="flex flex-col gap-2 sm:flex-row">
        <Select
          value={provider}
          onValueChange={(value) => {
            const next = value as ProviderName;
            onProviderChange(next);
            onModelChange(PROVIDERS[next].models[0]);
          }}
        >
          <SelectTrigger className="w-full sm:flex-1">
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
          <SelectTrigger className="w-full sm:flex-1">
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
          className="w-full sm:flex-1"
        />
      </div>

      <div className="flex items-center justify-between gap-2">
        <span className="text-xs text-muted-foreground">
          {config.free ? "Free tier available" : "Bring your own key"} — held in this browser
          tab only, never saved.
        </span>
        {apiKey && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => onApiKeyChange("")}
            className="h-7 shrink-0 text-xs"
          >
            Clear key
          </Button>
        )}
      </div>
    </div>
  );
}
