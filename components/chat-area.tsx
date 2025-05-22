"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Send,
  User,
  ChevronDown,
  Image,
  Upload,
  Loader2,
  X,
  PlusCircle,
  Search,
  Globe,
} from "lucide-react";
import { ChatMessage } from "@/components/chat-message";
import { useMobile } from "@/hooks/use-mobile";
import { ThemeToggle } from "@/components/theme-toggle";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Switch } from "@/components/ui/switch";

interface ChatAreaProps {
  onMenuClick: () => void;
  sidebarOpen: boolean;
  sidebarCollapsed?: boolean;
}

// 모델 타입 정의
type Model = {
  id: string;
  name: string;
};

// Message 타입 정의
interface Message {
  id: number;
  role: "user" | "system";
  content: string;
  imageUrl?: string; // 이미지 URL을 저장하기 위한 필드 추가
  citations?: Citation[];
}

// 인용 정보 타입 정의
interface Citation {
  url: string;
  title: string;
  start_index: number;
  end_index: number;
}

// 채팅 히스토리 타입 정의
interface ChatHistory {
  id: number;
  title: string;
  lastMessage: string;
  timestamp: Date;
}

// API URL 설정
const API_URL = "http://localhost:8001/api";

// 파일을 base64로 변환하는 유틸리티 함수
const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const base64String = reader.result as string;
      resolve(base64String); // 전체 data:image URL 반환
    };
    reader.onerror = (error) => reject(error);
  });
};

// blob을 base64로 변환하는 유틸리티 함수
const blobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onload = () => {
      const base64String = reader.result as string;
      const base64Data = base64String.split(",")[1]; // data:image/jpeg;base64, 부분 제거
      resolve(base64Data);
    };
    reader.onerror = (error) => reject(error);
  });
};

export function ChatArea({
  onMenuClick,
  sidebarOpen,
  sidebarCollapsed,
}: ChatAreaProps) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const isMobile = useMobile();
  const [streamingMessage, setStreamingMessage] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState(true);

  // 채팅 제목 관련 상태 추가
  const [chatTitle, setChatTitle] = useState<string>("Current Chat");
  const [isEditingTitle, setIsEditingTitle] = useState<boolean>(false);
  const titleInputRef = useRef<HTMLInputElement>(null);

  // 채팅 히스토리 상태 추가
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([]);
  const [currentChatId, setCurrentChatId] = useState<number>(1);

  // 이미지 업로드 관련 상태 추가
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // 입력창 참조 추가
  const inputRef = useRef<HTMLInputElement>(null);

  // 메시지 컨테이너 참조 생성
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // 상태 변화 추적을 위한 ref 추가
  const prevLoadingRef = useRef<boolean>(false);
  const prevStreamingRef = useRef<boolean>(false);

  // 무언가 표시 중인지 확인하는 ref 추가
  const resetInProgressRef = useRef<boolean>(false);

  // 채팅ID를 추적하는 상태 추가
  const [chatId, setChatId] = useState<number>(1);

  // 웹 검색 관련 상태 추가
  const [webSearchEnabled, setWebSearchEnabled] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isSearchMode, setIsSearchMode] = useState<boolean>(false);

  // 전역 chatHistory 상태를 공유하기 위한 이벤트 발송 함수
  const broadcastChatHistory = useCallback(() => {
    // 채팅 히스토리가 업데이트됨을 알리는 커스텀 이벤트 생성
    const event = new CustomEvent("chatHistoryUpdated", {
      detail: { chatHistory },
    });
    window.dispatchEvent(event);
  }, [chatHistory]);

  // 채팅 히스토리 업데이트 시 이벤트 발송
  useEffect(() => {
    broadcastChatHistory();
  }, [chatHistory, broadcastChatHistory]);

  // 전역 newChat 이벤트 리스너 추가
  useEffect(() => {
    // 이벤트 리스너 함수 생성
    const handleNewChatEvent = () => {
      console.log("New Chat 이벤트 감지됨!");
      startNewChat();
    };

    // 이벤트 리스너 등록
    window.addEventListener("newChat", handleNewChatEvent);

    // 컴포넌트 언마운트 시 이벤트 리스너 제거
    return () => {
      window.removeEventListener("newChat", handleNewChatEvent);
    };
  }, []); // 빈 의존성 배열로 컴포넌트 마운트/언마운트 시에만 실행

  // 채팅 전환 이벤트 리스너 추가
  useEffect(() => {
    // 이벤트 리스너 함수 생성
    const handleSwitchChatEvent = (event: CustomEvent) => {
      console.log("Switch Chat 이벤트 감지됨!", event.detail.chat);
      switchToChat(event.detail.chat);
    };

    // 이벤트 리스너 등록
    window.addEventListener(
      "switchChat",
      handleSwitchChatEvent as EventListener
    );

    // 컴포넌트 언마운트 시 이벤트 리스너 제거
    return () => {
      window.removeEventListener(
        "switchChat",
        handleSwitchChatEvent as EventListener
      );
    };
  }, []); // 빈 의존성 배열로 컴포넌트 마운트/언마운트 시에만 실행

  // 초기 인사말 설정을 위한 함수
  const showInitialGreeting = useCallback(() => {
    // 컴포넌트 마운트 시 초기 메시지 설정
    const initialMessage: Message = {
      id: 1,
      role: "system",
      content: "무엇을 도와드릴까요?",
    };

    // 스트리밍 효과를 위해 빈 메시지로 시작
    setMessages([{ ...initialMessage, content: "" }]);
    setIsStreaming(true);

    // 점진적으로 글자를 추가하여 스트리밍 효과 구현
    let text = "무엇을 도와드릴까요?";
    let currentIndex = 0;

    const interval = setInterval(() => {
      if (currentIndex <= text.length) {
        setMessages([
          { ...initialMessage, content: text.substring(0, currentIndex) },
        ]);
        currentIndex++;
      } else {
        setIsStreaming(false);
        clearInterval(interval);
      }
    }, 50);

    return interval;
  }, []);

  // 컴포넌트 마운트 시 초기 인사말 표시
  useEffect(() => {
    const interval = showInitialGreeting();
    return () => clearInterval(interval);
  }, [chatId, showInitialGreeting]);

  // AI 응답이 끝나면 입력창으로 포커스 이동
  useEffect(() => {
    // 이전 상태가 로딩 중이거나 스트리밍 중이었고, 현재는 둘 다 아닌 경우에만 포커스 설정
    const wasActive = prevLoadingRef.current || prevStreamingRef.current;
    const isNowInactive = !isLoading && !isStreaming;

    if (wasActive && isNowInactive && inputRef.current) {
      // 약간의 지연을 두고 포커스 설정 (UI 업데이트 후)
      setTimeout(() => {
        inputRef.current?.focus();
      }, 300);
    }

    // 현재 상태를 이전 상태로 저장
    prevLoadingRef.current = isLoading;
    prevStreamingRef.current = isStreaming;
  }, [isLoading, isStreaming]);

  // 스크롤을 맨 아래로 이동시키는 함수
  const scrollToBottom = (behavior: ScrollBehavior = "smooth") => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior, block: "start" });
    }
  };

  // 메시지가 추가되거나 업데이트될 때마다 스크롤 조정
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 스트리밍 상태가 변경될 때 스크롤 조정
  useEffect(() => {
    if (isStreaming) {
      const interval = setInterval(() => {
        if (chatContainerRef.current) {
          const isNearBottom =
            chatContainerRef.current.scrollHeight -
              chatContainerRef.current.scrollTop -
              chatContainerRef.current.clientHeight <
            100;

          if (isNearBottom) {
            scrollToBottom("auto");
          }
        }
      }, 100);

      return () => clearInterval(interval);
    }
  }, [isStreaming]);

  // 사용 가능한 모델 목록 - 프론트엔드에 고정
  const models: Model[] = [
    { id: "gpt-4.1", name: "GPT-4.1" },
    { id: "gpt-4o", name: "GPT-4o" },
    { id: "o4-mini", name: "O4-mini" },
    { id: "o3", name: "O3" },
  ];

  // 선택된 모델 상태
  const [selectedModel, setSelectedModel] = useState<Model>(models[0]);

  // 이미지가 화면에 표시되고 있는지 확인하는 함수
  const isImageVisible = (): boolean => {
    // 첨부된 이미지 확인 - 선택된 파일 또는 미리보기 URL이 있으면 이미지가 있는 것으로 간주
    if (selectedFile || previewUrl) {
      return true;
    }

    // 서버 사이드 렌더링 중에는 document가 없으므로 안전하게 처리
    if (typeof window === "undefined") {
      return false;
    }

    // 이미 화면에 표시된 첨부 이미지 확인
    const attachedImage = document.querySelector(
      'img[alt*="Uploaded image"]'
    ) as HTMLImageElement;
    return !!attachedImage;
  };

  // 파일 선택 핸들러 추가
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;

    if (file) {
      // 파일 타입 검증
      const validTypes = ["image/jpeg", "image/png", "image/gif", "image/webp"];
      if (!validTypes.includes(file.type)) {
        setError(
          "지원되지 않는 이미지 형식입니다. JPEG, PNG, GIF 또는 WEBP 형식만 지원합니다."
        );
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
        return;
      }

      // 파일 크기 검증 (20MB 제한)
      const maxSize = 20 * 1024 * 1024; // 20MB
      if (file.size > maxSize) {
        setError("이미지 크기가 너무 큽니다. 최대 20MB까지 지원합니다.");
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
        return;
      }

      console.log(
        `File selected: ${file.name}, type: ${file.type}, size: ${(
          file.size /
          1024 /
          1024
        ).toFixed(2)}MB`
      );
      setSelectedFile(file);
      setError(null);

      // 이미지 미리보기 생성
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result as string);
      };
      reader.onerror = () => {
        setError("이미지 파일을 읽는 중 오류가 발생했습니다.");
        setSelectedFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      };
      reader.readAsDataURL(file);
    }
  };

  // 이미지 분석 핸들러 수정
  const handleImageAnalysis = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // 이미지 파일 또는 URL 확인
      let imageData: string | null = null;
      let imageFile: File | null = null;

      // 1. 선택된 파일이 있는 경우
      if (selectedFile) {
        // 파일 형식 재확인
        const validTypes = [
          "image/jpeg",
          "image/png",
          "image/gif",
          "image/webp",
        ];
        if (!validTypes.includes(selectedFile.type)) {
          throw new Error(
            "지원되지 않는 이미지 형식입니다. JPEG, PNG, GIF 또는 WEBP 형식만 지원합니다."
          );
        }

        imageFile = selectedFile;
        try {
          imageData = await fileToBase64(selectedFile);
          console.log(
            `Debug - Selected file: ${selectedFile.name}, type: ${
              selectedFile.type
            }, size: ${(selectedFile.size / 1024 / 1024).toFixed(2)}MB`
          );
        } catch (e) {
          console.error("File to base64 conversion error:", e);
          throw new Error("이미지 변환 중 오류가 발생했습니다.");
        }
      }
      // 2. 미리보기 URL만 있는 경우
      else if (previewUrl) {
        imageData = previewUrl;
        console.log(`Debug - Using preview URL`);
      }
      // 3. 이미 화면에 표시된 첨부 이미지를 찾는 경우
      else if (typeof window !== "undefined") {
        const attachedImage = document.querySelector(
          'img[alt*="Uploaded image"]'
        ) as HTMLImageElement;

        if (attachedImage && attachedImage.src) {
          imageData = attachedImage.src;
          console.log(`Debug - Found attached image in DOM`);
        }
      }

      // 이미지가 없으면 에러
      if (!imageData) {
        throw new Error("분석할 이미지를 찾을 수 없습니다");
      }

      // 이미지 형식 확인
      if (!imageData.startsWith("data:image/")) {
        throw new Error(
          "이미지 형식이 올바르지 않습니다. data:image/ 형식이어야 합니다."
        );
      }

      // 이미지 형식 확인
      const imageFormat = imageData.split(";")[0].split(":")[1].toLowerCase();
      console.log(`Debug - Image format: ${imageFormat}`);

      // 지원되는 형식인지 확인
      const supportedFormats = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
      ];
      if (!supportedFormats.includes(imageFormat)) {
        console.warn(
          `Warning - Image format ${imageFormat} may not be supported`
        );
      }

      // 사용자 메시지 생성
      const userMessage: Message = {
        id: messages.length + 1,
        role: "user",
        content: input || "이 이미지를 분석해주세요.",
        imageUrl: imageData,
      };

      // 메시지 목록에 사용자 메시지 추가
      setMessages((prev) => [...prev, userMessage]);
      setInput("");

      // 시스템 메시지 추가 (빈 메시지로 시작)
      const newMessageId = messages.length + 2;
      const initialMessage: Message = {
        id: newMessageId,
        role: "system",
        content: "",
      };
      setMessages((prev) => [...prev, initialMessage]);
      setIsStreaming(true);

      // FormData 생성
      const formData = new FormData();
      formData.append(
        "prompt",
        input || "이 이미지에 대해 자세히 설명해주세요."
      );
      formData.append("model", selectedModel.id);
      formData.append("max_tokens", "1000");
      formData.append("detail", "auto");
      formData.append("stream", "true"); // 스트리밍 활성화

      // 이전 대화 컨텍스트 추가 (이미지 없는 메시지만)
      const conversationHistory = messages
        .filter((msg) => !msg.imageUrl)
        .map((msg) => ({
          role: msg.role,
          content: msg.content,
        }));
      formData.append(
        "conversation_history",
        JSON.stringify(conversationHistory)
      );

      // 이미지 추가: 파일이 있으면 파일로, 없으면 base64 문자열로
      if (imageFile) {
        console.log("Uploading as file:", imageFile.name, imageFile.type);
        formData.append("file", imageFile);
      } else {
        console.log("Uploading as base64 image");
        formData.append("base64_image", imageData);
      }

      console.log(`Debug - Sending request with model: ${selectedModel.id}`);

      // API 요청
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
        setError("API 요청 시간이 초과되었습니다. 다른 모델을 선택해보세요.");
        setIsLoading(false);
        setIsStreaming(false);
      }, 30000);

      try {
        const response = await fetch(`${API_URL}/upload-image`, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        clearTimeout(timeoutId); // 타임아웃 해제

        if (!response.ok) {
          let errorMessage = "이미지 분석 중 오류가 발생했습니다.";
          try {
            const errorData = await response.json();
            console.error("API error response:", errorData);
            errorMessage =
              errorData.detail || errorData.error?.message || errorMessage;
          } catch (e) {
            const errorText = await response.text();
            console.error("API error text:", errorText);
            errorMessage = errorText || errorMessage;
          }
          throw new Error(errorMessage);
        }

        // body가 없으면 에러
        if (!response.body) {
          throw new Error("응답 본문이 없습니다.");
        }

        // 스트리밍 응답 처리
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = "";
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          // 텍스트 디코딩
          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          // 이벤트 스트림 형식 처리 (data: {...}\n\n)
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.trim() === "") continue;
            if (line.includes("[DONE]")) {
              setIsStreaming(false);
              continue;
            }

            try {
              // "data: " 접두사 제거
              const jsonStr = line.replace(/^data: /, "").trim();
              if (!jsonStr) continue;

              const data = JSON.parse(jsonStr);

              if (data.content) {
                fullContent += data.content;

                // 실시간으로 메시지 업데이트
                setMessages((currentMessages) =>
                  currentMessages.map((msg) =>
                    msg.id === newMessageId
                      ? { ...msg, content: fullContent }
                      : msg
                  )
                );
              }

              if (data.is_streaming === false) {
                setIsStreaming(false);
              }
            } catch (e) {
              console.error("JSON 파싱 오류:", e, line);
            }
          }
        }
      } catch (error) {
        // AbortController에 의한 취소 확인
        if (error instanceof Error && error.name === "AbortError") {
          console.error("Request aborted due to timeout");
          throw new Error(
            "요청 시간이 초과되었습니다. 다른 모델을 선택해보세요."
          );
        }
        throw error; // 다른 오류는 상위 catch 블록으로 전달
      }

      // 요청이 완료되면 상태 초기화
      setSelectedFile(null);
      setPreviewUrl(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      console.error("이미지 분석 오류:", err);
      setError(
        err instanceof Error
          ? err.message
          : "이미지 분석 중 오류가 발생했습니다"
      );
      setIsStreaming(false);
    } finally {
      setIsLoading(false);
    }
  };

  // 메시지 전송 함수 수정 - 웹 검색 지원 추가
  const sendMessage = async () => {
    // 입력 없고 이미지도 없으면 아무 것도 하지 않음
    if (!input.trim() && !(selectedFile || previewUrl)) {
      return;
    }

    // 입력 저장 - API 호출 전에 입력값 저장
    const currentInput = input.trim();

    // 이미지가 화면에 표시되고 있는지 확인 (클라이언트 사이드에서만)
    const hasImage =
      selectedFile ||
      previewUrl ||
      (typeof window !== "undefined" &&
        !!document.querySelector('img[alt*="Uploaded image"]'));

    if (hasImage) {
      // 이미지가 있으면 이미지 분석 수행
      await handleImageAnalysis();
      return;
    }

    // 기존 텍스트 메시지 처리 로직
    const userMessage: Message = {
      id: messages.length + 1,
      role: "user",
      content: currentInput,
    };

    // 첫 메시지인 경우 채팅 히스토리에 추가
    if (messages.length <= 1) {
      // 초기 인사말만 있는 경우를 포함
      const newChat: ChatHistory = {
        id: currentChatId,
        title: chatTitle,
        lastMessage:
          currentInput.length > 30
            ? currentInput.substring(0, 30) + "..."
            : currentInput,
        timestamp: new Date(),
      };

      setChatHistory((prev) => [
        newChat,
        ...prev.filter((chat) => chat.id !== currentChatId),
      ]);
    } else {
      // 기존 채팅 업데이트
      setChatHistory((prev) =>
        prev.map((chat) =>
          chat.id === currentChatId
            ? {
                ...chat,
                lastMessage:
                  currentInput.length > 30
                    ? currentInput.substring(0, 30) + "..."
                    : currentInput,
                timestamp: new Date(),
              }
            : chat
        )
      );
    }

    // 메시지 목록에 사용자 메시지 추가
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);
    setStreamingMessage("");

    // 사용자 메시지가 추가된 후 스크롤 조정 (즉시)
    setTimeout(() => scrollToBottom("auto"), 50);

    const newMessageId = messages.length + 2;

    try {
      // 웹 검색 모드일 때 별도 처리
      if (webSearchEnabled) {
        // 시스템 메시지(응답) 추가
        const initialMessage: Message = {
          id: newMessageId,
          role: "system",
          content: "",
        };
        setMessages((prev) => [...prev, initialMessage]);
        setIsStreaming(true);

        // 웹 검색 API 호출
        const response = await fetch(`${API_URL}/websearch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: currentInput,
            model: selectedModel.id,
            temperature: 0.7,
            max_tokens: 1000,
          }),
        });

        if (!response.ok) {
          let errorMessage = "웹 검색 중 오류가 발생했습니다.";
          try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorMessage;
          } catch (e) {
            const errorText = await response.text();
            errorMessage = errorText || errorMessage;
          }
          throw new Error(errorMessage);
        }

        const data = await response.json();
        setMessages((currentMessages) =>
          currentMessages.map((msg) =>
            msg.id === newMessageId
              ? { ...msg, content: data.response }
              : msg
          )
        );
        setIsStreaming(false);
        setIsLoading(false);
        setIsSearchMode(false);
        return;
      }

      // 기존(일반) 채팅 API 호출
      // 실제 API 호출에 사용할 메시지 목록 생성 - 매번 새로 생성하여 이전 응답이 섞이지 않도록 함
      const filteredMessages = messages.filter(
        (msg) =>
          // 일반 모드에서는 이미지 URL이 있는 메시지와 웹 검색 관련 메시지만 제외
          !msg.imageUrl &&
          !(
            msg.role === "system" &&
            (msg.content.includes("이미지를") ||
              msg.content.includes("PwC") ||
              msg.content.includes("로고") ||
              msg.content.toLowerCase().includes("image") ||
              msg.content.includes("삼일회계법인"))
          )
      );

      // 사용자 메시지는 항상 추가 (가장 마지막 사용자 메시지)
      filteredMessages.push(userMessage);

      // API 호출 데이터 준비
      const requestData: any = {
        messages: filteredMessages.map((msg) => ({
          role: msg.role,
          content: msg.content,
        })),
        model: selectedModel.id,
        temperature: 0.7,
        max_tokens: 1000,
        stream: true,
      };

      // 빈 메시지로 시스템 응답 추가
      const initialMessage: Message = {
        id: newMessageId,
        role: "system",
        content: "",
      };
      setMessages((prev) => [...prev, initialMessage]);
      setIsStreaming(true);

      // POST 요청으로 스트리밍 데이터 가져오기
      const controller = new AbortController();
      const timeoutId = setTimeout(() => {
        controller.abort();
        setError("API 요청 시간이 초과되었습니다. 다른 모델을 선택해보세요.");
        setIsLoading(false);
        setIsStreaming(false);
      }, 30000);

      try {
        const response = await fetch(`${API_URL}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestData),
          signal: controller.signal,
        });

        clearTimeout(timeoutId); // 타임아웃 해제

        if (!response.ok) {
          throw new Error(
            `API 요청 실패: ${response.status} ${response.statusText}`
          );
        }

        if (!response.body) {
          throw new Error("응답 본문이 없습니다.");
        }

        // 응답 스트림 처리
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = "";
        let buffer = "";
        let citations: Citation[] = [];

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          // 텍스트 디코딩
          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          // 이벤트 스트림 형식 처리 (data: {...}\n\n)
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.trim() === "") continue;
            if (line.includes("[DONE]")) {
              setIsStreaming(false);
              continue;
            }

            try {
              // "data: " 접두사 제거
              const jsonStr = line.replace(/^data: /, "").trim();
              if (!jsonStr) continue;

              const data = JSON.parse(jsonStr);

              if (data.content) {
                fullContent += data.content;

                // 실시간으로 메시지 업데이트
                setMessages((currentMessages) =>
                  currentMessages.map((msg) =>
                    msg.id === newMessageId
                      ? { ...msg, content: fullContent }
                      : msg
                  )
                );
              }

              // 인용 정보가 있으면 추가
              if (data.citations) {
                citations = data.citations;
                console.log("인용 정보 수신:", citations.length);

                // 메시지 업데이트
                setMessages((currentMessages) =>
                  currentMessages.map((msg) =>
                    msg.id === newMessageId
                      ? {
                          ...msg,
                          content: fullContent,
                          citations: citations,
                        }
                      : msg
                  )
                );
              }

              if (data.is_streaming === false) {
                // 스트리밍 종료 시 최종 메시지 업데이트
                setMessages((currentMessages) =>
                  currentMessages.map((msg) =>
                    msg.id === newMessageId
                      ? {
                          ...msg,
                          content: fullContent,
                          citations:
                            citations.length > 0 ? citations : undefined,
                        }
                      : msg
                  )
                );
                setIsStreaming(false);
              }
            } catch (e) {
              console.error("JSON 파싱 오류:", e, line);
            }
          }
        }
      } catch (error) {
        // AbortController에 의한 취소 확인
        if (error instanceof Error && error.name === "AbortError") {
          console.error("Request aborted due to timeout");
          throw new Error(
            "요청 시간이 초과되었습니다. 다른 모델을 선택해보세요."
          );
        }
        throw error; // 다른 오류는 상위 catch 블록으로 전달
      }
    } catch (err) {
      console.error("API 에러:", err);
      setError(
        err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다."
      );
      setIsStreaming(false);
    } finally {
      setIsLoading(false);
    }
  };

  // 엔터 키로 메시지 전송
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // 파일 선택 버튼 클릭 핸들러
  const handleSelectClick = () => {
    fileInputRef.current?.click();
  };

  // 파일 제거 핸들러
  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // 모델 선택 핸들러
  const handleModelSelect = (model: Model) => {
    try {
      // 선택된 모델 업데이트
      setSelectedModel(model);
      // 에러 상태 초기화
      setError(null);

      // 디버깅용 로그
      console.log(`모델 변경됨: ${model.name} (${model.id})`);

      // AI가 응답할 준비가 되었음을 알려주는 메시지 추가
      setMessages((prev) => {
        // 이전 메시지가 있고 마지막 메시지가 사용자 메시지인 경우에만 추가
        if (prev.length > 0 && prev[prev.length - 1].role === "user") {
          return [
            ...prev,
            {
              id: prev.length + 1,
              role: "system",
              content: `모델이 ${model.name}(으)로 변경되었습니다.`,
            },
          ];
        }
        return prev;
      });
    } catch (err) {
      console.error("모델 선택 오류:", err);
    }
  };

  // 새로운 함수: 채팅 제목 편집 시작
  const startTitleEdit = () => {
    setIsEditingTitle(true);
    // 다음 렌더링 이후에 포커스 설정
    setTimeout(() => {
      if (titleInputRef.current) {
        titleInputRef.current.focus();
        titleInputRef.current.select();
      }
    }, 10);
  };

  // 새로운 함수: 채팅 제목 편집 완료
  const finishTitleEdit = () => {
    // 빈 제목이면 기본값으로 설정
    if (!chatTitle.trim()) {
      setChatTitle("새 대화");
    }
    setIsEditingTitle(false);
  };

  // 새로운 함수: 채팅 제목 변경 시 키보드 이벤트 처리
  const handleTitleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      finishTitleEdit();
    } else if (e.key === "Escape") {
      e.preventDefault();
      setIsEditingTitle(false);
    }
  };

  // 채팅 초기화 함수 수정 - 새 채팅 생성
  const startNewChat = () => {
    // 모든 상태 초기화
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    setStreamingMessage("");
    setInput("");
    setIsLoading(false);
    setIsStreaming(false);
    setMessages([]);
    setChatTitle("Current Chat");

    // 새 채팅 ID 생성
    const newChatId = Date.now();
    setCurrentChatId(newChatId);

    // chatId를 증가시켜 새로운 useEffect 트리거
    setChatId((prev) => prev + 1);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    // 포커스 설정
    setTimeout(() => {
      inputRef.current?.focus();
    }, 500);
  };

  // 채팅 전환 함수 추가
  const switchToChat = (chat: ChatHistory) => {
    // 현재 채팅 저장 로직 추가 필요 (필요한 경우)

    // 선택한 채팅으로 상태 업데이트
    setCurrentChatId(chat.id);
    setChatTitle(chat.title);

    // 선택한 채팅의 메시지 로드 로직 필요 (현재는 빈 구현)
    setMessages([
      {
        id: 1,
        role: "system",
        content: "이전 대화를 불러왔습니다. 계속 대화를 이어나가세요.",
      },
    ]);

    // 상태 초기화
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    setStreamingMessage("");
    setInput("");
    setIsLoading(false);
    setIsStreaming(false);

    // 포커스 설정
    setTimeout(() => {
      inputRef.current?.focus();
    }, 500);
  };

  // 웹 검색 토글 핸들러
  const toggleWebSearch = () => {
    setWebSearchEnabled((prev) => {
      // 웹 검색이 비활성화될 때는 검색 모드와 검색 쿼리도 초기화
      if (prev) {
        setIsSearchMode(false);
        setSearchQuery("");
      }
      // 대화를 초기화하지 않도록 수정 - 기존 대화 유지
      return !prev;
    });
  };

  // 검색 모드 토글 핸들러
  const toggleSearchMode = () => {
    // 검색 모드 토글
    setIsSearchMode((prev) => !prev);
    // 검색 모드가 켜지면 입력창에 포커스
    if (!isSearchMode && inputRef.current) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  };

  return (
    <div className="h-full flex flex-col bg-white dark:bg-background overflow-hidden">
      {/* Header */}
      <header className="border-b border-gray-200 dark:border-border py-2 px-3 flex items-center justify-between bg-white dark:bg-background shrink-0">
        {/* 왼쪽: 모델 선택/채팅 제목 */}
        <div className="flex items-center">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                className="flex items-center gap-2 h-9 border-gray-200 hover:border-gray-300 dark:border-gray-700 dark:hover:border-gray-600"
              >
                <span className="font-medium">모델: </span>
                <span className="text-gray-900 dark:text-white">
                  {selectedModel.name}
                </span>
                <ChevronDown size={16} />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56" sideOffset={5}>
              <div className="py-1">
                {models.map((model) => (
                  <DropdownMenuItem
                    key={model.id}
                    onClick={() => handleModelSelect(model)}
                    className={`mb-1 ${
                      selectedModel.id === model.id
                        ? "bg-gray-100 dark:bg-gray-800 font-medium"
                        : ""
                    }`}
                  >
                    {model.name}
                  </DropdownMenuItem>
                ))}
              </div>
            </DropdownMenuContent>
          </DropdownMenu>

          {isEditingTitle ? (
            <input
              ref={titleInputRef}
              type="text"
              value={chatTitle}
              onChange={(e) => setChatTitle(e.target.value)}
              onKeyDown={handleTitleKeyDown}
              onBlur={finishTitleEdit}
              className="ml-3 px-2 py-1 text-base font-medium bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500 dark:focus:ring-orange-600"
              maxLength={50}
            />
          ) : (
            <h2
              className="font-semibold ml-3 cursor-pointer hover:text-orange-500 dark:hover:text-orange-400 transition-colors"
              onClick={startTitleEdit}
              title="클릭하여 채팅 제목 변경"
            >
              {chatTitle}
            </h2>
          )}
        </div>

        {/* 오른쪽: 테마 토글/메뉴/유저 버튼 */}
        <div className="flex items-center gap-2">
          {/* 메뉴: 오른쪽에 배치, 더 작고 세련된 스타일 */}
          <nav className="ml-2">
            <ul className="flex space-x-3">
              <li>
                <a href="#" className="text-gray-500 dark:text-gray-300 hover:text-orange-500 dark:hover:text-orange-400 text-sm px-2 py-1 rounded transition-colors font-medium">문서번역</a>
              </li>
              <li>
                <a href="#" className="text-gray-500 dark:text-gray-300 hover:text-orange-500 dark:hover:text-orange-400 text-sm px-2 py-1 rounded transition-colors font-medium">계약서 검토</a>
              </li>
              <li>
                <a href="#" className="text-gray-500 dark:text-gray-300 hover:text-orange-500 dark:hover:text-orange-400 text-sm px-2 py-1 rounded transition-colors font-medium">AI Accountant</a>
              </li>
              <li>
                <a href="#" className="text-gray-500 dark:text-gray-300 hover:text-orange-500 dark:hover:text-orange-400 text-sm px-2 py-1 rounded transition-colors font-medium">Tax AI</a>
              </li>
            </ul>
          </nav>
          {/* 다크모드 토글 버튼 */}
          <ThemeToggle />
          {/* 유저 버튼 */}
          <Button
            variant="ghost"
            size="icon"
            className="h-9 w-9 rounded-full bg-gray-100 hover:bg-gray-200 dark:bg-secondary dark:hover:bg-secondary/80"
          >
            <User size={18} />
            <span className="sr-only">Login</span>
          </Button>
        </div>
      </header>

      {/* Chat messages */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-auto p-4 space-y-4 bg-white dark:bg-background"
      >
        {messages.map((message, index) => {
          // 이미지 표시 여부 결정:
          // 1. 현재 메시지가 사용자 메시지이고 이미지 URL이 있는지 확인
          // 2. 현재 메시지 이전에 동일한 이미지 URL을 가진 메시지가 있는지 확인
          // 3. 이전에 동일한 이미지가 없는 경우에만 이미지 표시
          const isFirstOccurrence =
            message.role === "user" &&
            !!message.imageUrl &&
            !messages
              .slice(0, index)
              .some(
                (prevMsg) =>
                  prevMsg.imageUrl === message.imageUrl &&
                  prevMsg.imageUrl !== undefined
              );

          return (
            <ChatMessage
              key={message.id}
              message={message}
              isStreaming={isStreaming && message.id === messages.length}
              showImage={isFirstOccurrence}
            />
          );
        })}
        {isLoading && !isStreaming && (
          <div className="text-center py-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              AI가 응답 중입니다...
            </span>
          </div>
        )}
        {error && (
          <div className="text-center py-3 px-4 my-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <span className="text-sm text-red-600 dark:text-red-400 font-medium">
              <span className="font-bold">오류:</span> {error}
            </span>
          </div>
        )}
        {/* 스크롤 위치를 조정하기 위한 참조 요소 */}
        <div ref={messagesEndRef} />
      </div>

      {/* 이미지 미리보기 영역 */}
      {previewUrl && (
        <div className="p-1 bg-gray-50 dark:bg-secondary/50 shrink-0">
          <div className="relative max-w-xs mx-auto">
            <img
              src={previewUrl}
              alt="Preview"
              className="h-28 mx-auto rounded-lg object-contain"
            />
            <button
              onClick={handleRemoveFile}
              className="absolute top-1 right-1 p-1 rounded-full bg-red-500 text-white hover:bg-red-700"
              aria-label="Remove image"
            >
              <X size={14} />
            </button>
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="p-3 bg-white dark:bg-background border-t border-gray-200 dark:border-border shrink-0">
        <div className="flex gap-2 max-w-3xl mx-auto">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/jpeg,image/png,image/webp,image/gif"
            className="hidden"
          />
          <div className="group relative">
            <Button
              type="button"
              size="icon"
              variant="outline"
              onClick={handleSelectClick}
              disabled={isLoading || isSearchMode}
              className="border-gray-200 dark:border-secondary hover:bg-gray-100 dark:hover:bg-secondary/80"
              title="이미지 업로드"
            >
              <Image size={18} />
              <span className="sr-only">Upload Image</span>
            </Button>
            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50">
              이미지 업로드 (최대 20MB)
            </div>
          </div>

          <div className="group relative">
            <Button
              type="button"
              size="icon"
              variant={webSearchEnabled ? "default" : "outline"}
              onClick={() => {
                toggleWebSearch();
                // 웹 검색이 활성화될 때 바로 검색 모드도 활성화
                if (!webSearchEnabled) {
                  setIsSearchMode(true);
                  // 웹 검색 활성화할 때 대화 초기화하지 않음 (제목만 변경)
                  setChatTitle("웹 검색 활성화됨");
                  setTimeout(() => {
                    inputRef.current?.focus();
                  }, 100);
                } else {
                  setIsSearchMode(false);
                  // 웹 검색이 비활성화될 때도 새 대화 시작하지 않고 현재 대화 유지
                  setChatTitle("일반 대화");
                }
              }}
              disabled={isLoading}
              className={`${
                webSearchEnabled
                  ? "bg-orange-500 text-white hover:bg-orange-600"
                  : "border-gray-200 dark:border-secondary hover:bg-gray-100 dark:hover:bg-secondary/80"
              }`}
              title={webSearchEnabled ? "웹 검색 끄기" : "웹 검색 켜기"}
            >
              <Globe size={18} />
              <span className="sr-only">Toggle Web Search</span>
            </Button>
            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50">
              {webSearchEnabled ? "웹 검색 끄기" : "웹 검색 켜기"}
            </div>
          </div>

          <Input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              webSearchEnabled
                ? "웹 검색어를 입력하세요..."
                : selectedFile
                ? "이미지에 대한 질문을 입력하세요..."
                : "Type your message..."
            }
            className="flex-1 border-gray-200 dark:bg-secondary dark:border-secondary focus-visible:ring-orange-500 focus-visible:border-orange-500"
            disabled={isLoading}
          />

          <Button
            type="submit"
            size="icon"
            onClick={() => {
              const hasImage = selectedFile || previewUrl;
              const hasAttachedImage =
                typeof window !== "undefined" &&
                !!document.querySelector('img[alt*="Uploaded image"]');

              if (hasImage || hasAttachedImage) {
                handleImageAnalysis();
              } else {
                sendMessage();
              }
            }}
            disabled={isLoading}
            className="bg-orange-500 hover:bg-orange-600 text-white relative group"
            title={webSearchEnabled ? "웹 검색 실행" : "메시지 전송"}
          >
            {isLoading ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Send size={18} />
            )}
            <span className="sr-only">Send</span>
            <div className="absolute bottom-full right-0 mb-2 p-2 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50">
              {webSearchEnabled ? "웹 검색 실행" : "메시지 전송"}
            </div>
          </Button>
        </div>
        {selectedFile && (
          <p className="text-xs text-center text-gray-500 dark:text-muted-foreground mt-2">
            선택된 이미지: {selectedFile.name} (
            {(selectedFile.size / (1024 * 1024)).toFixed(2)}MB)
          </p>
        )}
        {isSearchMode && (
          <p className="text-xs text-center text-orange-500 mt-2">
            웹 검색 모드: 검색어를 입력하고 검색 결과를 채팅에서 받아보세요
          </p>
        )}
        <p className="text-xs text-center text-gray-500 dark:text-muted-foreground mt-2">
          AI may produce inaccurate information about people, places, or facts.
        </p>
      </div>
    </div>
  );
}
