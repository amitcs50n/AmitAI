import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "Aevon",
  description: "Aevon private assistant interface.",
};

export const viewport: Viewport = {
  colorScheme: "dark",
  themeColor: "#0c0c0b",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
