"use client";

import { useState } from "react";
import { ProviderBar } from "@/components/provider-bar";
import { Assistant } from "@/components/assistant";
import { InventoryDashboard } from "@/components/inventory-dashboard";
import { ForecastTab } from "@/components/forecast-tab";
import { MlopsTab } from "@/components/mlops-tab";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DEFAULT_PROVIDER, PROVIDERS, type ProviderName } from "@/lib/providers";

export default function Home() {
  const [provider, setProvider] = useState<ProviderName>(DEFAULT_PROVIDER);
  const [model, setModel] = useState(PROVIDERS[DEFAULT_PROVIDER].models[0]);
  const [apiKey, setApiKey] = useState("");

  return (
    <main className="flex h-screen flex-col">
      <ProviderBar
        provider={provider}
        onProviderChange={setProvider}
        model={model}
        onModelChange={setModel}
        apiKey={apiKey}
        onApiKeyChange={setApiKey}
      />

      <Tabs defaultValue="assistant" className="flex min-h-0 flex-1 flex-col p-3">
        <TabsList>
          <TabsTrigger value="assistant">AI Assistant</TabsTrigger>
          <TabsTrigger value="inventory">Inventory Dashboard</TabsTrigger>
          <TabsTrigger value="forecast">Demand Forecast</TabsTrigger>
          <TabsTrigger value="mlops">MLOps Monitor</TabsTrigger>
        </TabsList>

        <TabsContent value="assistant" className="min-h-0 flex-1 pt-3">
          <Assistant provider={provider} model={model} apiKey={apiKey} />
        </TabsContent>
        <TabsContent value="inventory" className="min-h-0 flex-1 overflow-y-auto pt-3">
          <InventoryDashboard />
        </TabsContent>
        <TabsContent value="forecast" className="min-h-0 flex-1 overflow-y-auto pt-3">
          <ForecastTab />
        </TabsContent>
        <TabsContent value="mlops" className="min-h-0 flex-1 overflow-y-auto pt-3">
          <MlopsTab />
        </TabsContent>
      </Tabs>
    </main>
  );
}
