import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BC Wildfire Map",
  description: "Live BC wildfires, past seasons and where fires typically start",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 antialiased">{children}</body>
    </html>
  );
}
