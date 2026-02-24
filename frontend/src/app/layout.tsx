import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const jbMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MedScribe AI",
  description: "Enterprise Clinical Documentation Agent",
  icons: {
    icon: "/medscribe-logo.png",
    shortcut: "/medscribe-logo.png",
    apple: "/medscribe-logo.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} ${jbMono.variable} antialiased bg-background text-foreground min-h-screen overflow-hidden`}
      >
        {children}
      </body>
    </html>
  );
}
