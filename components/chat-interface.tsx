"use client";

import { useState } from "react";
import { Sidebar } from "@/components/sidebar";
import { ChatArea } from "@/components/chat-area";
import { useMobile } from "@/hooks/use-mobile";

export default function ChatInterface() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const isMobile = useMobile();

  return (
    <div className="flex h-screen bg-white dark:bg-background">
      {/* Sidebar - hidden on mobile when closed, or when collapsed on desktop */}
      <div
        className={`${sidebarOpen || !isMobile ? "block" : "hidden"} ${
          isMobile ? "fixed inset-0 z-50" : sidebarCollapsed ? "w-16" : "w-64"
        } transition-all duration-300 ease-in-out border-r border-gray-200 dark:border-border`}
      >
        <Sidebar
          onClose={() => setSidebarOpen(false)}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col bg-white dark:bg-background">
        <ChatArea
          onMenuClick={() =>
            isMobile
              ? setSidebarOpen(!sidebarOpen)
              : setSidebarCollapsed(!sidebarCollapsed)
          }
          sidebarOpen={sidebarOpen}
          sidebarCollapsed={sidebarCollapsed}
        />
      </div>
    </div>
  );
}
