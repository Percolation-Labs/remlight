/**
 * App - Root application component
 *
 * Sets up routing between chat and settings pages.
 */

import { BrowserRouter, Routes, Route } from "react-router-dom"
import { ChatPage, SettingsPage, AgentBuilderPage } from "@/pages"

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat/:sessionId" element={<ChatPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/agent-builder" element={<AgentBuilderPage />} />
        <Route path="/agent-builder/:agentName" element={<AgentBuilderPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
