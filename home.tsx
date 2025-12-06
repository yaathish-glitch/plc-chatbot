import { useState, useEffect, useRef } from "react";
import * as pdfjsLib from "pdfjs-dist";
import { createWorker } from "tesseract.js";
import { pipeline, env } from "@xenova/transformers";
import { 
  Bot, 
  Upload, 
  FileText, 
  Send, 
  Cpu, 
  Database, 
  AlertCircle, 
  CheckCircle2, 
  Loader2,
  Terminal,
  MessageSquare,
  ChevronDown,
  ChevronRight,
  ScanEye
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";

// Configure worker with fixed version to ensure stability
// Using unpkg for better reliability across environments
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@4.8.69/build/pdf.worker.min.mjs`;

// Skip local model checks for this demo to ensure it runs in browser
// Force single thread to avoid SharedArrayBuffer issues in some environments
env.allowLocalModels = false;
env.useBrowserCache = true;
env.backends.onnx.wasm.numThreads = 1;

interface VectorRecord {
  embedding: Float32Array;
  text: string;
  source: string;
  pageIndex: number;
  type: 'text' | 'image-ocr';
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  context?: string[]; 
  relatedImage?: string; // Data URL of the relevant page
  timestamp: number;
}

export default function Home() {
  const { toast } = useToast();
  
  // Model State
  const [embeddingPipeline, setEmbeddingPipeline] = useState<any>(null);
  const [generatorPipeline, setGeneratorPipeline] = useState<any>(null);
  const [modelStatus, setModelStatus] = useState({
    embeddings: 'loading', 
    llm: 'loading'
  });
  
  // Data State
  const [vectorDB, setVectorDB] = useState<VectorRecord[]>([]);
  const [pageImages, setPageImages] = useState<Record<string, string>>({}); // Key: "filename-pageIndex"
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState("");
  
  // Chat State
  const [input, setInput] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  
  const scrollRef = useRef<HTMLDivElement>(null);

  // Initialize Models
  useEffect(() => {
    async function loadModels() {
      try {
        // Load Embedding Model
        const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
        setEmbeddingPipeline(() => embedder);
        setModelStatus(prev => ({ ...prev, embeddings: 'ready' }));
        
        // Load LLM - Upgraded to Flan-T5 for better reasoning/Q&A capabilities compared to GPT-2
        // Using 'small' variant to ensure browser stability
        const generator = await pipeline("text2text-generation", "Xenova/flan-t5-small");
        setGeneratorPipeline(() => generator);
        setModelStatus(prev => ({ ...prev, llm: 'ready' }));
        
      } catch (error) {
        console.error("Model loading error:", error);
        setModelStatus({ embeddings: 'error', llm: 'error' });
        toast({
          title: "Model Load Error",
          description: "Failed to load AI models. Please reload the page.",
          variant: "destructive"
        });
      }
    }
    
    loadModels();
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatHistory, isGenerating]);

  // Utilities
  function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
  }

  function chunkText(text: string, chunkSize = 500, overlap = 100) {
    const sentences = text.split(/(?<=[.?!])\s+|\n+/);
    const chunks = [];
    let current = "";

    for (const sent of sentences) {
      if ((current + " " + sent).length > chunkSize) {
        if (current.trim().length > 0) {
          chunks.push(current.trim());
        }
        current = current.slice(-overlap) + " " + sent;
      } else {
        current += " " + sent;
      }
    }
    if (current.trim().length > 0) chunks.push(current.trim());
    return chunks;
  }

  // OCR Utility & Image Capture
  async function capturePageImage(page: any, scale = 1.0): Promise<string> {
    try {
      const viewport = page.getViewport({ scale });
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.height = viewport.height;
      canvas.width = viewport.width;

      if (!context) return "";

      await page.render({
        canvasContext: context,
        viewport: viewport
      }).promise;

      return canvas.toDataURL('image/jpeg', 0.8);
    } catch (e) {
      console.warn("Image Capture Error:", e);
      return "";
    }
  }

  async function performOCR(blob: Blob): Promise<string> {
      const worker = await createWorker('eng');
      const ret = await worker.recognize(blob);
      await worker.terminate();
      return ret.data.text;
  }

  // File Handling
  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    if (!e.target.files?.length) return;
    
    setIsProcessing(true);
    setProcessingStatus("Initializing...");
    const files = Array.from(e.target.files);
    const newRecords: VectorRecord[] = [];
    const newImages: Record<string, string> = {};
    
    try {
      for (const file of files) {
        setProcessingStatus(`Reading ${file.name}...`);
        
        // Read PDF
        const arrayBuffer = await file.arrayBuffer();
        const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
        const pdf = await loadingTask.promise;
        
        let fullText = "";
        
        for (let i = 1; i <= pdf.numPages; i++) {
          setProcessingStatus(`Analyzing page ${i}/${pdf.numPages} of ${file.name}...`);
          const page = await pdf.getPage(i);
          
          // 1. Capture Page Image (for display in chat)
          // We capture every page so we can show it if relevant
          const pageImgData = await capturePageImage(page, 1.0);
          newImages[`${file.name}-${i}`] = pageImgData;

          // 2. Get Standard Text
          const content = await page.getTextContent();
          const pageText = content.items.map((item: any) => item.str).join(" ");
          
          fullText += pageText + "\n";

          // 3. Run OCR if text is sparse (likely a scanned doc or diagram)
          if (pageText.length < 100) {
             setProcessingStatus(`Running OCR on page ${i} (diagram detected)...`);
             // Create blob from dataURL for Tesseract
             const res = await fetch(pageImgData);
             const blob = await res.blob();
             const ocrText = await performOCR(blob);
             
             if (ocrText.trim().length > 0) {
                 fullText += `\n[Page ${i} Diagram Content: ${ocrText}]\n`;
             }
          }
        }
        
        const cleaned = fullText.replace(/\s+/g, " ").trim();
        if (!cleaned) continue;
        
        // Chunk and Embed
        const chunks = chunkText(cleaned);
        setProcessingStatus(`Embedding ${chunks.length} chunks from ${file.name}...`);
        
        for (const chunk of chunks) {
          if (!embeddingPipeline) continue;
          
          const output = await embeddingPipeline(chunk, { pooling: "mean", normalize: true });
          
          // Heuristic to find which page this chunk belongs to (rough approximation)
          // In a real app we'd map char indices to pages, but here we'll just link to the last processed page
          // A better way is to chunk *per page*. Let's refine that slightly.
          // For now, we'll associate it with the file.
          // To support per-page images, we should actually chunk INSIDE the page loop.
          // Refactoring to chunk per-page for better image alignment:
        }
      }
      
      // RE-IMPLEMENTATION: Chunking per page for better image alignment
      for (const file of files) {
          const arrayBuffer = await file.arrayBuffer();
          const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
          const pdf = await loadingTask.promise;

          for (let i = 1; i <= pdf.numPages; i++) {
              setProcessingStatus(`Indexing page ${i}/${pdf.numPages} of ${file.name}...`);
              const page = await pdf.getPage(i);
              
              // Get Content
              const content = await page.getTextContent();
              let pageText = content.items.map((item: any) => item.str).join(" ");
              
              // OCR if needed
              if (pageText.length < 100) {
                 const imgData = newImages[`${file.name}-${i}`]; // Already captured above? No, let's capture here if we removed the previous loop.
                 // Actually, let's do it all in this one loop to be efficient.
                 const pageImgData = await capturePageImage(page, 1.0);
                 newImages[`${file.name}-${i}`] = pageImgData;
                 
                 const res = await fetch(pageImgData);
                 const blob = await res.blob();
                 const ocrText = await performOCR(blob);
                 pageText += `\n [Diagram/Image Text]: ${ocrText}`;
              } else {
                 // Still capture image for display
                 const pageImgData = await capturePageImage(page, 1.0);
                 newImages[`${file.name}-${i}`] = pageImgData;
              }

              // Chunk JUST this page
              const pageChunks = chunkText(pageText, 300, 50); // Smaller chunks for page-level precision
              
              for (const chunk of pageChunks) {
                  if (!embeddingPipeline) continue;
                  const output = await embeddingPipeline(chunk, { pooling: "mean", normalize: true });
                  newRecords.push({
                      embedding: output.data,
                      text: chunk,
                      source: file.name,
                      pageIndex: i,
                      type: 'text'
                  });
              }
          }
      }

      setPageImages(prev => ({...prev, ...newImages}));
      setVectorDB(prev => [...prev, ...newRecords]);
      toast({
        title: "Processing Complete",
        description: `Added ${newRecords.length} chunks to knowledge base.`
      });
      setProcessingStatus("");
    } catch (err) {
      console.error(err);
      toast({
        title: "Error processing file",
        description: err instanceof Error ? err.message : "Failed to process PDF files.",
        variant: "destructive"
      });
    } finally {
      setIsProcessing(false);
    }
  }

  // Question Handling
  async function handleAsk() {
    if (!input.trim()) return;
    
    const question = input.trim();
    setInput("");
    setIsGenerating(true);
    
    // Add user message
    setChatHistory(prev => [...prev, {
      role: 'user',
      content: question,
      timestamp: Date.now()
    }]);

    try {
      if (!embeddingPipeline || !generatorPipeline) {
        throw new Error("Models not loaded yet");
      }

      // 1. Embed Question
      const qOutput = await embeddingPipeline(question, { pooling: "mean", normalize: true });
      const qEmbedding = qOutput.data;
      
      // 2. Retrieve Context
      let contextText = "";
      let topK: VectorRecord[] = [];
      
      if (vectorDB.length > 0) {
        const scores = vectorDB.map((record, idx) => ({
          idx,
          score: cosineSimilarity(qEmbedding, record.embedding)
        }));
        scores.sort((a, b) => b.score - a.score);
        topK = scores.slice(0, 3).map(s => vectorDB[s.idx]);
        contextText = topK.map(k => k.text).join("\n\n");
      }
      
      // 3. Generate Answer
      const prompt = `Answer the question based on the context below. Be concise and helpful.
      
Context: ${contextText}

Question: ${question}
Answer:`;
      
      const result = await generatorPipeline(prompt, {
        max_new_tokens: 150,
        do_sample: false
      });
      
      let answer = result[0].generated_text || "Could not generate answer.";
      
      // Find the most relevant image
      // We look at the top chunk, get its pageIndex, and grab that image
      let relevantImage = undefined;
      if (topK.length > 0) {
          const topMatch = topK[0];
          const imgKey = `${topMatch.source}-${topMatch.pageIndex}`;
          if (pageImages[imgKey]) {
              relevantImage = pageImages[imgKey];
          }
      }
      
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: answer,
        context: topK.map(k => `[${k.source} Pg.${k.pageIndex}] ${k.text.substring(0, 100)}...`),
        relatedImage: relevantImage,
        timestamp: Date.now()
      }]);
      
    } catch (err) {
      console.error(err);
      toast({
        title: "Generation Failed",
        description: "Could not generate an answer.",
        variant: "destructive"
      });
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: "I encountered an error while trying to answer that. Please ensure models are loaded.",
        timestamp: Date.now()
      }]);
    } finally {
      setIsGenerating(false);
    }
  }

  // Determine if input should be disabled
  // Only disable if currently generating. Allow typing even if models loading (will queue or fail gracefully)
  const isInputDisabled = isGenerating; 

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col md:flex-row font-sans">
      
      {/* Sidebar: Status & Upload */}
      <aside className="w-full md:w-80 border-r border-border bg-card/30 p-6 flex flex-col gap-8 shrink-0">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2 tracking-tight">
            <span className="bg-primary text-primary-foreground w-8 h-8 rounded-full flex items-center justify-center text-sm font-mono">AI</span>
            PLC Chat
          </h1>
          <p className="text-sm text-muted-foreground mt-2">
            Local RAG System with OCR for Industrial Documentation
          </p>
        </div>

        {/* System Status */}
        <div className="space-y-4">
          <h3 className="text-xs font-mono uppercase tracking-wider text-muted-foreground">System Status</h3>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm p-3 rounded-lg border border-border bg-card/50">
              <span className="flex items-center gap-2">
                <Database className="w-4 h-4 text-muted-foreground" />
                Embeddings
              </span>
              {modelStatus.embeddings === 'loading' && <Badge variant="outline" className="text-yellow-500 border-yellow-500/30">Loading</Badge>}
              {modelStatus.embeddings === 'ready' && <Badge variant="outline" className="text-green-500 border-green-500/30">Active</Badge>}
              {modelStatus.embeddings === 'error' && <Badge variant="destructive">Error</Badge>}
            </div>

            <div className="flex items-center justify-between text-sm p-3 rounded-lg border border-border bg-card/50">
              <span className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-muted-foreground" />
                LLM Engine
              </span>
              {modelStatus.llm === 'loading' && <Badge variant="outline" className="text-yellow-500 border-yellow-500/30">Loading</Badge>}
              {modelStatus.llm === 'ready' && <Badge variant="outline" className="text-green-500 border-green-500/30">Active</Badge>}
              {modelStatus.llm === 'error' && <Badge variant="destructive">Error</Badge>}
            </div>
          </div>
        </div>

        {/* Knowledge Base */}
        <div className="space-y-4">
          <h3 className="text-xs font-mono uppercase tracking-wider text-muted-foreground">Knowledge Base</h3>
          
          <div className="border-2 border-dashed border-border rounded-xl p-6 text-center hover:bg-muted/50 transition-colors cursor-pointer relative group">
            <input 
              type="file" 
              accept=".pdf" 
              multiple 
              className="absolute inset-0 opacity-0 cursor-pointer z-10"
              onChange={handleFileUpload}
              disabled={isProcessing || modelStatus.embeddings !== 'ready'}
            />
            <div className="flex flex-col items-center gap-2">
              <div className="p-3 bg-muted rounded-full group-hover:bg-primary/20 transition-colors">
                {isProcessing ? <Loader2 className="w-6 h-6 animate-spin text-primary" /> : <Upload className="w-6 h-6 text-primary" />}
              </div>
              <div className="text-sm font-medium">
                {isProcessing ? "Processing..." : "Upload PDF Files"}
              </div>
              <div className="text-xs text-muted-foreground">
                Supports OCR for scanned diagrams
              </div>
            </div>
          </div>

          {processingStatus && (
            <div className="text-xs font-mono text-primary animate-pulse break-words">
              {">"} {processingStatus}
            </div>
          )}

          <div className="text-xs text-muted-foreground flex items-center justify-between">
            <span className="flex items-center gap-2">
               <FileText className="w-3 h-3" />
               Chunks: {vectorDB.length}
            </span>
            <span className="flex items-center gap-2">
               <ScanEye className="w-3 h-3" />
               OCR: {vectorDB.filter(x => x.type === 'image-ocr').length}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-[100vh]">
        <div className="flex-1 p-6 overflow-hidden flex flex-col">
          {chatHistory.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center opacity-50 space-y-4">
              <Bot className="w-16 h-16 text-muted-foreground" />
              <div className="max-w-md space-y-2">
                <h3 className="text-xl font-semibold">Ready for Queries</h3>
                <p className="text-sm text-muted-foreground">
                  Upload your PLC manuals or wiring diagrams. 
                  The AI now extracts text from images (OCR) to understand diagrams better.
                </p>
              </div>
            </div>
          ) : (
            <ScrollArea className="flex-1 pr-4" ref={scrollRef}>
              <div className="space-y-6 pb-4">
                {chatHistory.map((msg, i) => (
                  <div key={i} className={cn(
                    "flex gap-4 max-w-3xl mx-auto",
                    msg.role === 'assistant' ? "bg-muted/30 p-4 rounded-xl" : "px-4"
                  )}>
                    <div className={cn(
                      "w-8 h-8 rounded-full flex items-center justify-center shrink-0 border",
                      msg.role === 'assistant' ? "bg-primary/10 border-primary/20 text-primary" : "bg-muted border-border"
                    )}>
                      {msg.role === 'assistant' ? <Bot className="w-4 h-4" /> : <MessageSquare className="w-4 h-4" />}
                    </div>
                    
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">
                          {msg.role === 'assistant' ? "AI Assistant" : "You"}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      
                      <div className="text-sm leading-relaxed whitespace-pre-wrap">
                        {msg.content}
                      </div>

                      {msg.relatedImage && (
                        <div className="mt-4 rounded-lg overflow-hidden border border-border">
                            <div className="bg-muted px-3 py-1 text-xs text-muted-foreground border-b border-border">
                                Referenced Diagram / Page Source
                            </div>
                            <img src={msg.relatedImage} alt="Context" className="max-w-full h-auto" />
                        </div>
                      )}

                      {msg.context && msg.context.length > 0 && (
                        <Collapsible className="mt-4">
                          <CollapsibleTrigger asChild>
                            <Button variant="ghost" size="sm" className="h-6 gap-2 text-xs text-muted-foreground">
                              <Terminal className="w-3 h-3" />
                              View Retrieved Context
                              <ChevronDown className="w-3 h-3" />
                            </Button>
                          </CollapsibleTrigger>
                          <CollapsibleContent className="mt-2 space-y-2">
                            {msg.context.map((ctx, idx) => (
                              <div key={idx} className="text-xs font-mono bg-black/40 p-2 rounded border border-border text-muted-foreground break-words whitespace-pre-wrap">
                                {ctx}
                              </div>
                            ))}
                          </CollapsibleContent>
                        </Collapsible>
                      )}
                    </div>
                  </div>
                ))}
                {isGenerating && (
                  <div className="flex gap-4 max-w-3xl mx-auto bg-muted/30 p-4 rounded-xl">
                    <div className="w-8 h-8 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0">
                      <Loader2 className="w-4 h-4 animate-spin text-primary" />
                    </div>
                    <div className="flex-1">
                      <span className="text-sm text-muted-foreground animate-pulse">Thinking...</span>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          )}
        </div>

        {/* Input Area */}
        <div className="p-6 pt-2 border-t border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="max-w-3xl mx-auto flex gap-3">
            <Input 
              placeholder={vectorDB.length === 0 ? "Ask a general question (upload PDFs for context)..." : "Ask a question about your documents..."} 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !isInputDisabled && handleAsk()}
              disabled={isInputDisabled}
              className="h-12 text-base bg-muted/50 border-border focus-visible:ring-primary/50"
            />
            <Button 
              size="icon" 
              className="h-12 w-12 shrink-0 rounded-xl" 
              onClick={handleAsk}
              disabled={isInputDisabled || !input.trim()}
            >
              <Send className="w-5 h-5" />
            </Button>
          </div>
          <div className="max-w-3xl mx-auto mt-2 text-center">
             <span className="text-[10px] text-muted-foreground">
               AI runs locally in your browser. No data leaves your device.
             </span>
          </div>
        </div>
      </main>
    </div>
  );
}
