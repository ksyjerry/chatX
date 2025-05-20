import { Avatar } from "@/components/ui/avatar";
import { User, Bot, Copy, Check, Link } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import rehypeHighlight from "rehype-highlight";
import React, { useState, useRef, useCallback, ReactNode } from "react";

interface Citation {
  url: string;
  title: string;
  start_index: number;
  end_index: number;
}

interface Message {
  id: number;
  role: "user" | "system";
  content: string;
  imageUrl?: string;
  citations?: Citation[];
}

interface ChatMessageProps {
  message: Message;
  isStreaming?: boolean;
  showImage?: boolean;
}

// 커스텀 컴포넌트 인터페이스
interface PreProps {
  children?: ReactNode;
  className?: string;
  [key: string]: any;
}

export function ChatMessage({
  message,
  isStreaming = false,
  showImage = false,
}: ChatMessageProps) {
  const isUser = message.role === "user";
  const [copiedMap, setCopiedMap] = useState<Record<string, boolean>>({});
  const [showCitations, setShowCitations] = useState<boolean>(true);

  // 코드 복사 함수
  const copyToClipboard = useCallback((text: string, id: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedMap((prev) => ({ ...prev, [id]: true }));
      setTimeout(() => {
        setCopiedMap((prev) => ({ ...prev, [id]: false }));
      }, 2000);
    });
  }, []);

  // 코드 내용을 추출하는 함수
  const extractCodeText = useCallback((node: any): string => {
    if (!node) return "";

    // 문자열인 경우 바로 반환
    if (typeof node === "string") return node;

    // props와 children이 있는 객체인 경우 재귀적으로 처리
    if (typeof node === "object") {
      // props.children이 있는 경우
      if ("props" in node && node.props && node.props.children) {
        const children = node.props.children;

        // children이 배열인 경우
        if (Array.isArray(children)) {
          return children.map(extractCodeText).join("");
        }

        // children이 객체나 문자열인 경우
        return extractCodeText(children);
      }
    }

    return "";
  }, []);

  // pre 요소를 커스텀 컴포넌트로 대체
  const PreBlock = useCallback(
    ({ children, ...props }: PreProps) => {
      const preRef = useRef<HTMLPreElement>(null);
      const codeId = `code-${Math.random().toString(36).substr(2, 9)}`;

      // 코드 텍스트 추출
      const codeText = extractCodeText(children);
      const isCopied = copiedMap[codeId] || false;

      return (
        <div className="code-block-wrapper">
          <button
            className={`code-copy-button ${isCopied ? "copied" : ""}`}
            onClick={() => copyToClipboard(codeText, codeId)}
            aria-label="코드 복사"
            title="코드 복사"
          >
            <span className="flex items-center gap-1">
              {isCopied ? (
                <>
                  <Check size={14} />
                  <span className="text-xs">복사됨</span>
                </>
              ) : (
                <>
                  <Copy size={14} />
                  <span className="text-xs">복사</span>
                </>
              )}
            </span>
          </button>
          <pre ref={preRef} {...props}>
            {children}
          </pre>
        </div>
      );
    },
    [copiedMap, copyToClipboard, extractCodeText]
  );

  // 표 커스텀 컴포넌트
  const TableComponent = useCallback(({ children, ...props }: any) => {
    return (
      <div className="overflow-x-auto my-4">
        <table className="border-collapse w-full" {...props}>
          {children}
        </table>
      </div>
    );
  }, []);

  const TableHead = useCallback(({ children, ...props }: any) => {
    return (
      <thead className="bg-gray-100 dark:bg-gray-800" {...props}>
        {children}
      </thead>
    );
  }, []);

  const TableRow = useCallback(({ children, ...props }: any) => {
    return (
      <tr
        className="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors"
        {...props}
      >
        {children}
      </tr>
    );
  }, []);

  const TableCell = useCallback(({ children, ...props }: any) => {
    return (
      <td
        className="py-2 px-4 border-x border-gray-200 dark:border-gray-700"
        {...props}
      >
        {children}
      </td>
    );
  }, []);

  const TableHeader = useCallback(({ children, ...props }: any) => {
    return (
      <th
        className="py-3 px-4 text-left font-medium border-x border-gray-200 dark:border-gray-700"
        {...props}
      >
        {children}
      </th>
    );
  }, []);

  // 인용 정보 표시 토글
  const toggleCitations = useCallback(() => {
    setShowCitations((prev) => !prev);
  }, []);

  return (
    <div className={`flex gap-4 ${isUser ? "flex-row-reverse" : ""}`}>
      <Avatar
        className={`flex items-center justify-center flex-shrink-0 ${
          isUser ? "bg-orange-500" : "bg-gray-700"
        }`}
      >
        {isUser ? (
          <User className="h-5 w-5 text-white" />
        ) : (
          <Bot className="h-5 w-5 text-white" />
        )}
      </Avatar>
      <div
        className={`flex-1 flex items-start ${
          isUser ? "justify-end" : "justify-start"
        } flex-col ${isUser ? "items-end" : "items-start"}`}
      >
        {isUser ? (
          <div className="text-gray-700 dark:text-foreground whitespace-pre-line text-right">
            {message.imageUrl && showImage && (
              <div
                className={`mb-2 ${isUser ? "ml-auto" : "mr-auto"} max-w-xs`}
              >
                <img
                  src={message.imageUrl}
                  alt="Uploaded image"
                  className="rounded-lg object-contain max-h-64 border border-gray-200 dark:border-gray-700 shadow-sm"
                />
              </div>
            )}
            {message.content}
          </div>
        ) : (
          <div className="w-full">
            <div
              className={`text-gray-700 dark:text-foreground prose dark:prose-invert 
              prose-headings:font-semibold
              prose-a:text-orange-500 dark:prose-a:text-orange-400
              prose-a:hover:text-orange-600 dark:prose-a:hover:text-orange-300
              prose-p:my-2
              prose-li:my-0.5
              prose-pre:bg-gray-100 dark:prose-pre:bg-gray-800 prose-pre:border prose-pre:border-gray-200 dark:prose-pre:border-gray-700
              prose-pre:shadow-sm
              prose-table:border-collapse prose-table:w-full
              prose-th:bg-gray-100 dark:prose-th:bg-gray-800 prose-th:p-2 prose-th:text-left
              prose-td:border prose-td:p-2 prose-td:border-gray-200 dark:prose-td:border-gray-700
              max-w-none ${isUser ? "text-right" : "text-left"}`}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw, rehypeHighlight]}
                components={{
                  pre: PreBlock,
                  table: TableComponent,
                  thead: TableHead,
                  tr: TableRow,
                  td: TableCell,
                  th: TableHeader,
                  strong: ({ node, children, ...props }) => (
                    <strong className="font-bold" {...props}>
                      {children}
                    </strong>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
              {isStreaming && <span className="animate-pulse">▌</span>}
            </div>

            {/* 인용 정보 표시 부분 */}
            {message.citations && message.citations.length > 0 && (
              <div className="mt-3 w-full">
                <div className="flex items-center gap-2">
                  <button
                    onClick={toggleCitations}
                    className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 hover:text-orange-500 dark:hover:text-orange-400"
                  >
                    <Link size={14} />
                    {showCitations ? "인용 정보 숨기기" : "인용 정보 보기"} (
                    {message.citations.length})
                  </button>
                </div>

                {showCitations && (
                  <div className="mt-1 space-y-1 text-sm text-gray-600 dark:text-gray-300">
                    {message.citations.map((citation, index) => (
                      <div
                        key={index}
                        className="flex items-start gap-2 py-1 px-2 bg-gray-50 dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700"
                      >
                        <span className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                          [{index + 1}]
                        </span>
                        <div className="flex-1">
                          <a
                            href={citation.url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-orange-500 dark:text-orange-400 hover:underline"
                          >
                            {citation.title || citation.url}
                          </a>
                          <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                            {citation.url}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
