/**
 * MessageContent - Markdown rendering for message text
 *
 * Uses react-markdown with syntax highlighting for code blocks.
 */

import ReactMarkdown from "react-markdown"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism"
import type { Components } from "react-markdown"

interface MessageContentProps {
  content: string
}

/**
 * Custom components for react-markdown.
 */
const components: Components = {
  code({ className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || "")
    const isInline = !match

    if (isInline) {
      return (
        <code
          className="bg-zinc-100 text-zinc-800 px-1 py-0.5 rounded text-xs font-mono"
          {...props}
        >
          {children}
        </code>
      )
    }

    return (
      <SyntaxHighlighter
        style={oneDark}
        language={match[1]}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: "0.375rem",
          fontSize: "0.75rem",
        }}
      >
        {String(children).replace(/\n$/, "")}
      </SyntaxHighlighter>
    )
  },
  p({ children }) {
    return <p className="mb-2 last:mb-0">{children}</p>
  },
  ul({ children }) {
    return <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>
  },
  ol({ children }) {
    return <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>
  },
  li({ children }) {
    return <li className="text-sm">{children}</li>
  },
  a({ href, children }) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-zinc-600 underline hover:text-zinc-900"
      >
        {children}
      </a>
    )
  },
  blockquote({ children }) {
    return (
      <blockquote className="border-l-2 border-zinc-300 pl-3 italic text-zinc-600 my-2">
        {children}
      </blockquote>
    )
  },
  h1({ children }) {
    return <h1 className="text-lg font-semibold mb-2">{children}</h1>
  },
  h2({ children }) {
    return <h2 className="text-base font-semibold mb-2">{children}</h2>
  },
  h3({ children }) {
    return <h3 className="text-sm font-semibold mb-1">{children}</h3>
  },
  pre({ children }) {
    return <pre className="my-2">{children}</pre>
  },
}

export function MessageContent({ content }: MessageContentProps) {
  if (!content || content.trim() === "") {
    return <span className="text-zinc-400 text-sm italic">No content</span>
  }

  return (
    <div className="prose prose-sm prose-zinc max-w-none text-sm leading-relaxed">
      <ReactMarkdown components={components}>{content}</ReactMarkdown>
    </div>
  )
}
