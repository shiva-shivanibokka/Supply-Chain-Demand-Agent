"use client";

import { Assistant } from "@/components/assistant";
import { InventoryDashboard } from "@/components/inventory-dashboard";
import { ForecastTab } from "@/components/forecast-tab";
import { MlopsTab } from "@/components/mlops-tab";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// 2.5in side gutter on wide screens, scaling down gracefully on smaller ones.
const GUTTER = "px-5 md:px-10 xl:px-[2.5in]";

function Logo() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="brandLogo" x1="0" y1="0" x2="34" y2="34" gradientUnits="userSpaceOnUse">
          <stop stopColor="#22d3ee" />
          <stop offset="0.5" stopColor="#6366f1" />
          <stop offset="1" stopColor="#d946ef" />
        </linearGradient>
      </defs>
      {/* hexagonal supply node */}
      <path
        d="M17 2.6 L28.4 9.1 L28.4 24.9 L17 31.4 L5.6 24.9 L5.6 9.1 Z"
        stroke="url(#brandLogo)"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      {/* forecast line rising through the node */}
      <path
        d="M9 22 L14 17 L18 20 L24.5 11"
        stroke="url(#brandLogo)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="24.5" cy="11" r="2.3" fill="url(#brandLogo)" />
    </svg>
  );
}

export default function Home() {
  return (
    <main className="flex h-screen flex-col">
      <header className="border-b border-border bg-card/40">
        <div className={`flex items-center gap-3 py-3.5 ${GUTTER}`}>
          <Logo />
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-foreground">
              Supply Chain Demand Agent <span className="brand-text">&amp; Forecaster</span>
            </h1>
            <p className="text-xs text-muted-foreground">
              Agentic inventory risk, demand forecasting, and RAG over supply-chain policy.
            </p>
          </div>
        </div>
      </header>

      <div className={`flex min-h-0 flex-1 flex-col py-4 ${GUTTER}`}>
        <Tabs defaultValue="inventory" className="flex min-h-0 flex-1 flex-col">
          <TabsList variant="line" className="h-auto gap-2 bg-transparent p-0">
            <TabsTrigger value="inventory" className="tab-pill after:hidden">
              Inventory Dashboard
            </TabsTrigger>
            <TabsTrigger value="forecast" className="tab-pill after:hidden">
              Demand Forecast
            </TabsTrigger>
            <TabsTrigger value="assistant" className="tab-pill after:hidden">
              AI Assistant
            </TabsTrigger>
            <TabsTrigger value="mlops" className="tab-pill after:hidden">
              MLOps Monitor
            </TabsTrigger>
          </TabsList>

          <TabsContent value="inventory" className="min-h-0 flex-1 overflow-y-auto pt-4">
            <InventoryDashboard />
          </TabsContent>
          <TabsContent value="forecast" className="min-h-0 flex-1 overflow-y-auto pt-4">
            <ForecastTab />
          </TabsContent>
          <TabsContent value="assistant" className="min-h-0 flex-1 pt-4">
            <Assistant />
          </TabsContent>
          <TabsContent value="mlops" className="min-h-0 flex-1 overflow-y-auto pt-4">
            <MlopsTab />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}
