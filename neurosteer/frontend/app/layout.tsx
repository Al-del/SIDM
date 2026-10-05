import type { Metadata, Viewport } from "next";
import { Inter_Tight, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const sans = Inter_Tight({ subsets: ["latin"], weight: ["300", "400", "500", "600"], variable: "--font-sans" });
const mono = JetBrains_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Neurosteer · EEG-steered language model",
  description:
    "Closed-loop EEG → LLM: Qwen answers one sentence at a time while you read, and a RAG-Mosaic decoder turns each EEG epoch into semantic vectors that steer the next sentence.",
  applicationName: "Neurosteer",
  keywords: ["EEG", "brain-computer interface", "LLM steering", "Qwen", "RAG-Mosaic", "neural decoding"],
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: "#040506",
  colorScheme: "dark",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
