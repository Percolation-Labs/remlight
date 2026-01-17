/**
 * App - Root application component
 *
 * Sets up routing between chat and settings pages.
 */

import { BrowserRouter, Routes, Route } from "react-router-dom"
import { ChatPage, SettingsPage } from "@/pages"

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/chat/:sessionId" element={<ChatPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
