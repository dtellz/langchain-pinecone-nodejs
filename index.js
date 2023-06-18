import { PineconeClient } from "@pinecone-database/pinecone";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import * as dotenv from "dotenv";
import { createPineconeIndex } from "./1-createPineconeIndex.js";
import { updatePinecone } from "./2-updatePinecone.js";
import { queryPineconeVectorStoreAndQueryLLM } from "./3-queryPineconeAndQueryGPT.js";
import path from "path";

dotenv.config();

const loader = new DirectoryLoader("./file-to-read", {
    ".json": (path) => new TextLoader(path)
}); 

const docs = await loader.load();

const question = "What is the test stack this project uses?";
const indexName = "read-index";
const VectorDimension = 1536; // OpenAI embeddings model has 1536 dimensions

const client = new PineconeClient();
await client.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
});

(async () => {
    await createPineconeIndex(client, indexName, VectorDimension);
    await updatePinecone(client, indexName, docs);
    await queryPineconeVectorStoreAndQueryLLM(client, indexName, question);
})
