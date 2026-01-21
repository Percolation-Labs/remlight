import { test, expect } from "@playwright/test"

/**
 * Agent Builder E2E tests.
 *
 * These tests verify the agent builder functionality including:
 * - Page navigation and layout
 * - Schema preview panel sections
 * - Chat panel interaction
 * - Sidebar integration
 */

test.describe("Agent Builder Page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/agent-builder")
  })

  test("should display the agent builder page", async ({ page }) => {
    // Check for page title
    await expect(page.getByText("Create New Agent")).toBeVisible()

    // Check for back button
    await expect(page.getByRole("button", { name: "Back" })).toBeVisible()

    // Check for save button
    await expect(page.getByRole("button", { name: /save agent/i })).toBeVisible()
  })

  test("should have split layout with schema preview and chat", async ({ page }) => {
    // Schema preview panel (left side)
    await expect(page.getByRole("heading", { name: "New Agent", exact: true })).toBeVisible()

    // Tools section
    await expect(page.getByRole("heading", { name: "Tools" })).toBeVisible()

    // System Prompt section
    await expect(page.getByRole("heading", { name: "System Prompt" })).toBeVisible()

    // Output Schema section
    await expect(page.getByRole("heading", { name: "Output Schema" })).toBeVisible()

    // Chat panel (right side)
    await expect(page.getByText("Agent Builder", { exact: true })).toBeVisible()
  })

  test("should navigate back to home when clicking Back", async ({ page }) => {
    await page.getByRole("button", { name: "Back" }).click()

    // Should be on the home page
    await expect(page).toHaveURL("/")
  })
})

test.describe("Schema Preview Panel", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/agent-builder")
  })

  test("should display empty tools section", async ({ page }) => {
    // Tools section should show empty state
    await expect(
      page.getByText("No tools configured. Add tools to give your agent capabilities.")
    ).toBeVisible()
  })

  test("should have Add button in tools section", async ({ page }) => {
    // Find the Add button near the Tools heading (traverse up to header container)
    const addButton = page.getByRole("heading", { name: "Tools" }).locator("../..").getByRole("button", { name: "Add" })
    await expect(addButton).toBeVisible()
  })

  test("should display empty system prompt section", async ({ page }) => {
    // System prompt should show placeholder
    await expect(
      page.getByText("Click to add a system prompt...")
    ).toBeVisible()
  })

  test("should have Edit button in system prompt section", async ({ page }) => {
    await expect(page.getByRole("button", { name: "Edit" })).toBeVisible()
  })

  test("should display empty output schema section", async ({ page }) => {
    // Should show empty state
    await expect(
      page.getByText("No output schema defined")
    ).toBeVisible()
  })

  test("should have Add Field button in output schema section", async ({ page }) => {
    await expect(page.getByRole("button", { name: "Add Field" })).toBeVisible()
  })

  test("should display metadata section", async ({ page }) => {
    await expect(page.getByText("Metadata")).toBeVisible()
    await expect(page.getByText("Name")).toBeVisible()
    await expect(page.getByText("Version")).toBeVisible()
  })
})

test.describe("System Prompt Editing", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/agent-builder")
  })

  test("should enter edit mode when clicking Edit button", async ({ page }) => {
    // Click Edit button
    await page.getByRole("button", { name: "Edit" }).click()

    // Should show textarea
    await expect(
      page.getByPlaceholder("Enter the system prompt for your agent...")
    ).toBeVisible()
  })

  test("should save system prompt when pressing Cmd+Enter", async ({ page }) => {
    // Click Edit button
    await page.getByRole("button", { name: "Edit" }).click()

    // Type in textarea
    const textarea = page.getByPlaceholder("Enter the system prompt for your agent...")
    await textarea.fill("You are a helpful assistant.")

    // Press Cmd+Enter to save
    await textarea.press("Meta+Enter")

    // Should exit edit mode and show the content
    await expect(page.getByText("You are a helpful assistant.")).toBeVisible()
  })

  test("should cancel edit when pressing Escape", async ({ page }) => {
    // Click Edit button
    await page.getByRole("button", { name: "Edit" }).click()

    // Type in textarea
    const textarea = page.getByPlaceholder("Enter the system prompt for your agent...")
    await textarea.fill("Some text that should be discarded")

    // Press Escape to cancel
    await textarea.press("Escape")

    // Should exit edit mode and show placeholder
    await expect(
      page.getByText("Click to add a system prompt...")
    ).toBeVisible()
  })
})

test.describe("Chat Panel", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/agent-builder")
  })

  test("should display welcome message", async ({ page }) => {
    await expect(
      page.getByText("I'm the Agent Builder")
    ).toBeVisible()
  })

  test("should have chat input", async ({ page }) => {
    await expect(
      page.getByPlaceholder("Describe your agent or ask a question...")
    ).toBeVisible()
  })

  test("should allow typing in chat input", async ({ page }) => {
    const input = page.getByPlaceholder("Describe your agent or ask a question...")

    await input.fill("I want to build an agent that analyzes feedback")

    await expect(input).toHaveValue("I want to build an agent that analyzes feedback")
  })
})

test.describe("Sidebar Integration", () => {
  test("should navigate to agent builder from sidebar", async ({ page }) => {
    // Start at home page
    await page.goto("/")

    // Click on Schema Builder icon in the nav rail
    const schemaBuilderIcon = page.locator('button[title="Schema Builder"]')
    await schemaBuilderIcon.click()

    // Click on "New Agent" or "Open Builder" button
    const newAgentButton = page.getByText("New Agent")
    if (await newAgentButton.isVisible()) {
      await newAgentButton.click()
    } else {
      const openBuilderButton = page.getByText("Open Builder")
      if (await openBuilderButton.isVisible()) {
        await openBuilderButton.click()
      }
    }

    // Should navigate to agent builder
    await expect(page).toHaveURL("/agent-builder")
  })
})

test.describe("Agent Builder with Agent Name", () => {
  test("should display agent name in title when editing existing agent", async ({ page }) => {
    // Navigate to agent builder with agent name
    await page.goto("/agent-builder/query-agent")

    // Should show agent name in title
    await expect(page.getByText("Edit: query-agent")).toBeVisible()
  })
})
