import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI English Conversation - Voice Practice",
  description: "Practice English conversation with AI using speech recognition and synthesis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className="antialiased"
      >
        {children}
      </body>
    </html>
  );
}
