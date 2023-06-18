import { PineconeClient } from "@pinecone-database/pinecone";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import * as dotenv from "dotenv";
import { updatePinecone } from "./updatePinecone.js";
import { queryPineconeVectorStoreAndQueryLLM } from "./queryPineconeAndQueryGPT.js";
import { createPineconeIndex } from "./createPineConeIndex.js";
import path from "path";

dotenv.config();

console.log("Starting...");
let docs = [];
try {
    const loader = new DirectoryLoader("./file-to-read", {
        ".json": (path) => new TextLoader(path)
    }); 
    docs = await loader.load();
    
} catch (err) {
    console.log(err);
}
console.log("Finished loading documents");



const question = "Tell me all the dependencies related to tests that this project has";
const indexName = "read-index";
const VectorDimension = 1536; // OpenAI embeddings model has 1536 dimensions

const client = new PineconeClient();
await client.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
});

(async () => {
    console.log("Starting...");
    await createPineconeIndex(client, indexName, VectorDimension);
    await updatePinecone(client, indexName, docs);
    await queryPineconeVectorStoreAndQueryLLM(client, indexName, question);
})();
