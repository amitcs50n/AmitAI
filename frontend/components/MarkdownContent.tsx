"use client";

import { Children, isValidElement, useState, type HTMLAttributes, type ReactNode } from "react";
import { Check, Copy } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

interface MarkdownContentProps {
  content: string;
  wrapCode?: boolean;
}

function CodeBlock({ code, language, wrap }: { code: string; language: string; wrap: boolean }) {
  const [copied, setCopied] = useState(false);

  async function copyCode() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="my-5 overflow-hidden rounded-xl border border-[#6e5139]/55 bg-[#090a0a]">
      <div className="flex items-center justify-between border-b border-[#6e5139]/35 px-4 py-2 text-[0.68rem] uppercase tracking-[0.14em] text-[#9f958b]">
        <span>{language || "code"}</span>
        <button
          aria-label="Copy code"
          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[#b9afa6] transition hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
          onClick={copyCode}
          type="button"
        >
          {copied ? <Check aria-hidden="true" className="h-3.5 w-3.5" /> : <Copy aria-hidden="true" className="h-3.5 w-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre
        className={cn(
          "overflow-x-auto p-4 text-[0.84rem] leading-6 text-[#e0d8d0]",
          wrap && "whitespace-pre-wrap break-words",
        )}
      >
        <code>{code}</code>
      </pre>
    </div>
  );
}

function extractCode(children: ReactNode): { code: string; language: string } {
  const child = Children.count(children) === 1 ? Children.only(children) : null;
  if (isValidElement<{ className?: string; children?: ReactNode }>(child)) {
    const className = child.props.className ?? "";
    return {
      code: String(child.props.children ?? "").replace(/\n$/, ""),
      language: className.match(/language-([\w-]+)/)?.[1] ?? "",
    };
  }
  return { code: String(children ?? "").replace(/\n$/, ""), language: "" };
}

export function MarkdownContent({ content, wrapCode = false }: MarkdownContentProps) {
  const components: Components = {
    pre({ children }: HTMLAttributes<HTMLPreElement>) {
      const { code, language } = extractCode(children);
      return <CodeBlock code={code} language={language} wrap={wrapCode} />;
    },
    code({ className, children }) {
      return (
        <code
          className={cn(
            "rounded border border-[#6e5139]/45 bg-[#191510] px-1.5 py-0.5 font-mono text-[0.88em] text-[#e5b489]",
            className,
          )}
        >
          {children}
        </code>
      );
    },
    a({ children, ...props }) {
      return (
        <a
          className="text-[#dfa573] underline decoration-[#8e613e] underline-offset-4 hover:text-[#f0c59f]"
          rel="noreferrer"
          target="_blank"
          {...props}
        >
          {children}
        </a>
      );
    },
    table({ children }) {
      return (
        <div className="my-5 overflow-x-auto rounded-xl border border-[#6e5139]/45">
          <table className="w-full border-collapse text-left text-sm">{children}</table>
        </div>
      );
    },
    th({ children }) {
      return <th className="border-b border-[#6e5139]/45 bg-[#17130f] px-3 py-2 font-medium text-[#e8dfd7]">{children}</th>;
    },
    td({ children }) {
      return <td className="border-b border-[#46372b]/50 px-3 py-2 align-top">{children}</td>;
    },
  };

  return (
    <div className="amitai-markdown space-y-4 text-[0.98rem] leading-7 text-[#ded9d4] [&_blockquote]:border-l-2 [&_blockquote]:border-[#a26e46] [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:text-[#aaa29b] [&_h1]:font-serif [&_h1]:text-2xl [&_h1]:text-[#f1ebe5] [&_h2]:font-serif [&_h2]:text-xl [&_h2]:text-[#f1ebe5] [&_h3]:text-lg [&_h3]:font-semibold [&_h3]:text-[#eee8e1] [&_hr]:border-[#6e5139]/45 [&_li]:my-1 [&_ol]:list-decimal [&_ol]:space-y-1 [&_ol]:pl-6 [&_p]:leading-7 [&_strong]:font-semibold [&_strong]:text-[#f1ebe5] [&_ul]:list-disc [&_ul]:space-y-1 [&_ul]:pl-6">
      <ReactMarkdown components={components} remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
