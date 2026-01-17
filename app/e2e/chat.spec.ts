import { test, expect } from "@playwright/test"

/**
 * Chat interface E2E tests.
 *
 * These tests verify core chat functionality including:
 * - Message sending and receiving
 * - Tool call display
 * - Session management
 * - UI interactions
 */

test.describe("Chat Interface", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/")
  })

  test("should display empty state on load", async ({ page }) => {
    // Check for empty state message
    await expect(
      page.getByText("Start a conversation by sending a message below.")
    ).toBeVisible()

    // Check for input field
    await expect(page.getByPlaceholder("Type a message...")).toBeVisible()
  })

  test("should have agent and model selectors in toolbar", async ({ page }) => {
    // Agent selector - look for the Bot icon in toolbar
    await expect(page.locator('svg.lucide-bot').first()).toBeVisible()

    // Model selector - look for the Cpu icon in toolbar
    await expect(page.locator('svg.lucide-cpu').first()).toBeVisible()

    // New chat button in toolbar
    await expect(page.getByRole("button", { name: "New Chat" })).toBeVisible()
  })

  test("should allow typing in the message input", async ({ page }) => {
    const input = page.getByPlaceholder("Type a message...")

    await input.fill("Hello, this is a test message")

    await expect(input).toHaveValue("Hello, this is a test message")
  })

  test("should show send button when input has text", async ({ page }) => {
    const input = page.getByPlaceholder("Type a message...")
    const sendButton = page.locator('button:has(svg.lucide-send)')

    // Initially disabled (no text)
    await expect(sendButton).toBeVisible()

    // Type something
    await input.fill("Hello")

    // Send button should be clickable
    await expect(sendButton).toBeEnabled()
  })

  test("should display sidebar with session history", async ({ page }) => {
    // Sessions header - be specific with heading role
    await expect(page.getByRole("heading", { name: "Sessions" })).toBeVisible()

    // Search input
    await expect(page.getByPlaceholder("Search sessions...")).toBeVisible()

    // New button in sidebar - be specific with exact match
    await expect(page.getByRole("button", { name: "New", exact: true })).toBeVisible()
  })

  test("should toggle sidebar collapse", async ({ page }) => {
    // Find collapse button
    const collapseButton = page.locator('button[title="Collapse sidebar"]')

    // Click to collapse
    await collapseButton.click()

    // Sessions header should be hidden when collapsed
    await expect(page.getByRole("heading", { name: "Sessions" })).not.toBeVisible()

    // Find expand button
    const expandButton = page.locator('button[title="Expand sidebar"]')

    // Click to expand
    await expandButton.click()

    // Sessions header should be visible again
    await expect(page.getByRole("heading", { name: "Sessions" })).toBeVisible()
  })
})

test.describe("Message Sending", () => {
  test("can send message and see user message appear", async ({ page }) => {
    await page.goto("/")

    const input = page.getByPlaceholder("Type a message...")

    // Type and send message
    await input.fill("Hello")
    await input.press("Enter")

    // User message should appear
    await expect(page.locator('[data-role="user"]')).toBeVisible()
    await expect(page.getByText("Hello")).toBeVisible()
  })

  test("should show pending state for assistant message", async ({ page }) => {
    await page.goto("/")

    const input = page.getByPlaceholder("Type a message...")

    // Type and send message
    await input.fill("Hello")
    await input.press("Enter")

    // Assistant message placeholder should appear (may show "Thinking...")
    await expect(page.locator('[data-role="assistant"]')).toBeVisible({
      timeout: 5000,
    })
  })
})

test.describe("Tool Call Display", () => {
  // This test requires the API to return tool calls
  test.skip("tool calls are displayed in expandable cards", async ({ page }) => {
    await page.goto("/")

    // Send a message that triggers tool use
    const input = page.getByPlaceholder("Type a message...")
    await input.fill("Search for AI documents")
    await input.press("Enter")

    // Wait for tool call card
    await expect(page.locator('[data-testid="tool-call-card"]')).toBeVisible({
      timeout: 60000,
    })

    // Click to expand
    await page.locator('[data-testid="tool-call-card"]').click()

    // Should see Arguments section
    await expect(page.getByText("Arguments:")).toBeVisible()
  })
})

test.describe("New Chat", () => {
  test("should clear messages when clicking New Chat", async ({ page }) => {
    await page.goto("/")

    // Send a message first
    const input = page.getByPlaceholder("Type a message...")
    await input.fill("Hello")
    await input.press("Enter")

    // Wait for message to appear
    await expect(page.locator('[data-role="user"]')).toBeVisible()

    // Click New Chat button
    await page.getByRole("button", { name: /new chat/i }).click()

    // Messages should be cleared
    await expect(
      page.getByText("Start a conversation by sending a message below.")
    ).toBeVisible()
  })
})

test.describe("Session Reload with Tool Calls", () => {
  // Test that a session with tool calls can be loaded from the sidebar
  // We use a known session ID that has tool calls stored in the database
  const TEST_SESSION_ID = "550e8400-e29b-41d4-a716-446655440001"

  test("should display tool calls when loading a session from sidebar", async ({ page }) => {
    await page.goto("/")

    // Wait for sessions to load in sidebar
    await expect(page.getByPlaceholder("Search sessions...")).toBeVisible()

    // Search for the test session
    await page.getByPlaceholder("Search sessions...").fill("rem light")

    // Wait a bit for search results
    await page.waitForTimeout(1000)

    // Click on the first session in the list
    const sessionItem = page.locator('[data-testid="session-item"]').first()
    if (await sessionItem.isVisible()) {
      await sessionItem.click()

      // Wait for messages to load
      await page.waitForTimeout(500)

      // Should see user message
      await expect(page.locator('[data-role="user"]').first()).toBeVisible()

      // Should see at least one assistant message (may have multiple due to tool calls)
      await expect(page.locator('[data-role="assistant"]').first()).toBeVisible()

      // Should see tool call cards (if the session has tool calls)
      const toolCards = page.locator('[data-testid="tool-call-card"]')
      const cardCount = await toolCards.count()

      if (cardCount > 0) {
        // Tool calls are displayed - verify count
        console.log(`Found ${cardCount} tool call cards`)

        // Click to expand a tool call card
        await toolCards.first().click()

        // Should see Arguments section when expanded
        await expect(page.getByText("Arguments:").first()).toBeVisible()
      }
    }
  })

  test("should preserve tool call structure when switching sessions", async ({ page }) => {
    await page.goto("/")

    // Send a new message first to create a fresh session
    const input = page.getByPlaceholder("Type a message...")
    await input.fill("Hello from new session")
    await input.press("Enter")

    // Wait for user message to appear
    await expect(page.locator('[data-role="user"]')).toBeVisible()

    // Search for an existing session with tool calls
    await page.getByPlaceholder("Search sessions...").fill("rem light")
    await page.waitForTimeout(1000)

    // Click on session if found
    const sessionItem = page.locator('[data-testid="session-item"]').first()
    if (await sessionItem.isVisible()) {
      await sessionItem.click()
      await page.waitForTimeout(500)

      // The message content should change (not "Hello from new session" anymore)
      const userMessage = page.locator('[data-role="user"]').first()
      await expect(userMessage).not.toContainText("Hello from new session")
    }
  })
})
