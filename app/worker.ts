import { ChatWindowMessage } from "@/schema/ChatWindowMessage";

import { Voy as VoyClient } from "voy-search";

import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";

import { WebPDFLoader } from "langchain/document_loaders/web/pdf";

import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { VoyVectorStore } from "@langchain/community/vectorstores/voy";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
  PromptTemplate,
} from "@langchain/core/prompts";
import { RunnableSequence, RunnablePick } from "@langchain/core/runnables";
import {
  AIMessage,
  type BaseMessage,
  HumanMessage,
} from "@langchain/core/messages";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import type { LanguageModelLike } from "@langchain/core/language_models/base";

import { LangChainTracer } from "@langchain/core/tracers/tracer_langchain";
import { Client } from "langsmith";

import { ChatOllama } from "@langchain/community/chat_models/ollama";

const embeddings = new HuggingFaceTransformersEmbeddings({
  modelName: "nomic-ai/nomic-embed-text-v1",
});

const voyClient = new VoyClient();
const vectorstore = new VoyVectorStore(voyClient, embeddings);

const OLLAMA_RESPONSE_SYSTEM_TEMPLATE = `
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.
Anything between the following \`context\` html blocks is retrieved from a knowledge bank, not part of the conversation with the user.
<context>
{context}
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.`;

const embedPDF = async (pdfBlob: Blob) => {
  const pdfLoader = new WebPDFLoader(pdfBlob, { parsedItemSeparator: " " });
  const docs = await pdfLoader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 50,
  });

  const splitDocs = await splitter.splitDocuments(docs);

  // Define regular expressions for Goals, Actions, and Metrics
  const goalRegex = /Goal\s+\d+:\s+([\s\S]*?)(?=Goal\s+\d+:|$)/gis;
  const actionRegex =
    /Actions?\s*:\s*([\s\S]*?)(?=Actions?\s*:\s*|Metrics?\s*:\s*|Goal\s+\d+:|$)/gis;
  const metricRegex =
    /Metrics?\s*:\s*([\s\S]*?)(?=Metrics?\s*:\s*|Actions?\s*:\s*|Goal\s+\d+:|$)/gis;

  const extractedContent: {
    goal: string;
    description: string;
    actions: string[];
    metrics: string[];
  }[] = [];

  for (const doc of splitDocs) {
    const content = doc.pageContent;
    let match;

    while ((match = goalRegex.exec(content)) !== null) {
      const goalContent = match[1];
      const [goalTitle, ...goalDescription] = goalContent
        .split("\n")
        .map((line) => line.trim());
      const goal = goalTitle;
      const description = goalDescription.join(" ").trim();

      // Extract Actions for the current Goal
      const actionMatches: string[] = [];
      let actionMatch;
      while ((actionMatch = actionRegex.exec(goalContent)) !== null) {
        actionMatches.push(actionMatch[1].trim());
      }

      // Extract Metrics for the current Goal
      const metricMatches: string[] = [];
      let metricMatch;
      while ((metricMatch = metricRegex.exec(goalContent)) !== null) {
        metricMatches.push(metricMatch[1].trim());
      }

      extractedContent.push({
        goal,
        description,
        actions: actionMatches,
        metrics: metricMatches,
      });
    }
  }

  // Log the extracted content for debugging
  self.postMessage({
    type: "log",
    data: extractedContent,
  });

  // Create Document instances from the extracted content
  const documents: Document<Record<string, any>>[] = extractedContent.map(
    (info) => {
      const pageContent = `Goal: ${info.goal}\nDescription: ${
        info.description
      }\nActions:\n${info.actions.join("\n")}\nMetrics:\n${info.metrics.join(
        "\n",
      )}`;
      return {
        pageContent,
        metadata: {},
      };
    },
  );

  // Add the documents to the vector store
  await vectorstore.addDocuments(documents);
};

const _formatChatHistoryAsMessages = async (
  chatHistory: ChatWindowMessage[],
) => {
  return chatHistory.map((chatMessage) => {
    if (chatMessage.role === "human") {
      return new HumanMessage(chatMessage.content);
    } else {
      return new AIMessage(chatMessage.content);
    }
  });
};

const queryVectorStore = async (
  messages: ChatWindowMessage[],
  {
    chatModel,
    devModeTracer,
  }: {
    chatModel: LanguageModelLike;
    devModeTracer?: LangChainTracer;
  },
) => {
  const text = messages[messages.length - 1].content;
  const chatHistory = await _formatChatHistoryAsMessages(messages.slice(0, -1));

  const responseChainPrompt = ChatPromptTemplate.fromMessages<{
    context: string;
    chat_history: BaseMessage[];
    question: string;
  }>([
    ["system", OLLAMA_RESPONSE_SYSTEM_TEMPLATE],
    new MessagesPlaceholder("chat_history"),
    ["user", `{input}`],
  ]);

  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: responseChainPrompt,
    documentPrompt: PromptTemplate.fromTemplate(
      `<doc>\n{page_content}\n</doc>`,
    ),
  });

  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a natural language search query to look up in order to get information relevant to the conversation. Do not respond with anything except the query.",
    ],
  ]);

  const historyAwareRetrieverChain = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever: vectorstore.asRetriever(),
    rephrasePrompt: historyAwarePrompt,
  });

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever: historyAwareRetrieverChain,
  });

  const fullChain = RunnableSequence.from([
    retrievalChain,
    new RunnablePick("answer"),
  ]);

  const stream = await fullChain.stream(
    {
      input: text,
      chat_history: chatHistory,
    },
    {
      callbacks: devModeTracer !== undefined ? [devModeTracer] : [],
    },
  );

  for await (const chunk of stream) {
    if (chunk) {
      self.postMessage({
        type: "chunk",
        data: chunk,
      });
    }
  }

  self.postMessage({
    type: "complete",
    data: "OK",
  });
};

// Listen for messages from the main thread
self.addEventListener("message", async (event: { data: any }) => {
  self.postMessage({
    type: "log",
    data: `Received data!`,
  });

  let devModeTracer;
  if (
    event.data.DEV_LANGCHAIN_TRACING !== undefined &&
    typeof event.data.DEV_LANGCHAIN_TRACING === "object"
  ) {
    devModeTracer = new LangChainTracer({
      projectName: event.data.DEV_LANGCHAIN_TRACING.LANGCHAIN_PROJECT,
      client: new Client({
        apiKey: event.data.DEV_LANGCHAIN_TRACING.LANGCHAIN_API_KEY,
      }),
    });
  }

  if (event.data.pdf) {
    try {
      await embedPDF(event.data.pdf);
    } catch (e: any) {
      self.postMessage({
        type: "error",
        error: e.message,
      });
      throw e;
    }
  } else {
    const modelConfig = event.data.modelConfig;
    const chatModel = new ChatOllama(modelConfig);
    try {
      await queryVectorStore(event.data.messages, {
        devModeTracer,
        chatModel,
      });
    } catch (e: any) {
      self.postMessage({
        type: "error",
        error: `${e.message}. Make sure you are running Ollama.`,
      });
      throw e;
    }
  }

  self.postMessage({
    type: "complete",
    data: "OK",
  });
});
