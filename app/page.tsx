"use client";

import { Assistant } from "@/components/assistant";
import { InventoryDashboard } from "@/components/inventory-dashboard";
import { ForecastTab } from "@/components/forecast-tab";
import { MlopsTab } from "@/components/mlops-tab";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Home() {
  return (
    <main className="flex h-screen flex-col">
      <header className="border-b border-border bg-card/40 px-4 py-3">
        <h1 className="text-lg font-semibold tracking-tight text-foreground">
          Supply Chain Demand Agent &amp; Forecaster
        </h1>
        <p className="text-xs text-muted-foreground">
          Agentic inventory risk, demand forecasting, and RAG over supply-chain policy.
        </p>
      </header>

      <Tabs defaultValue="inventory" className="flex min-h-0 flex-1 flex-col p-3">
        <TabsList>
          <TabsTrigger value="inventory">Inventory Dashboard</TabsTrigger>
          <TabsTrigger value="forecast">Demand Forecast</TabsTrigger>
          <TabsTrigger value="assistant">AI Assistant</TabsTrigger>
          <TabsTrigger value="mlops">MLOps Monitor</TabsTrigger>
        </TabsList>

        <TabsContent value="inventory" className="min-h-0 flex-1 overflow-y-auto pt-3">
          <InventoryDashboard />
        </TabsContent>
        <TabsContent value="forecast" className="min-h-0 flex-1 overflow-y-auto pt-3">
          <ForecastTab />
        </TabsContent>
        <TabsContent value="assistant" className="min-h-0 flex-1 pt-3">
          <Assistant />
        </TabsContent>
        <TabsContent value="mlops" className="min-h-0 flex-1 overflow-y-auto pt-3">
          <MlopsTab />
        </TabsContent>
      </Tabs>
    </main>
  );
}
