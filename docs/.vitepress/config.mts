import { defineConfig } from "vitepress";

export default defineConfig({
  title: "BikoDB",
  description: "Multi-model graph runtime in Rust with documented scope and maturity",
  base: "/BikoDB/",
  lastUpdated: true,
  sitemap: {
    hostname: "https://unnamedlab.github.io/BikoDB",
    lastmodDateOnly: false,
  },
  markdown: {
    image: {
      lazyLoading: true,
    },
  },
  ignoreDeadLinks: [/^https?:\/\/localhost/],
  head: [["link", { rel: "icon", href: "/BikoDB/logo.png" }]],
  themeConfig: {
    logo: "/logo.png",
    search: {
      provider: "local",
    },
    editLink: {
      pattern: "https://github.com/unnamedlab/BikoDB/tree/main/docs/:path",
      text: "Edit this page on GitHub",
    },
    nav: [
      { text: "Home", link: "/" },
      { text: "Guide", link: "/guide/" },
      { text: "Architecture", link: "/architecture/" },
      { text: "Benchmarks", link: "/benchmarks/" },
      { text: "Operations", link: "/architecture/operations-observability" },
      { text: "FAQ", link: "/faq" },
    ],
    sidebar: {
      "/": [
        {
          text: "Guide",
          items: [
            { text: "What is BikoDB?", link: "/guide/" },
            { text: "Installation", link: "/guide/installation" },
            { text: "Quickstart", link: "/guide/quickstart" },
            { text: "Supported Features & Limitations", link: "/guide/supported-features" },
          ],
        },
        {
          text: "Architecture",
          items: [
            { text: "Overview", link: "/architecture/" },
            { text: "Transactions & Concurrency", link: "/architecture/transactions" },
            { text: "Multi-model Consistency", link: "/architecture/multi-model-consistency" },
            { text: "Supported Guarantees", link: "/architecture/supported-guarantees" },
            { text: "Operations & Observability", link: "/architecture/operations-observability" },
          ],
        },
        {
          text: "Benchmarks",
          items: [
            { text: "Overview", link: "/benchmarks/" },
            { text: "Methodology", link: "/benchmarks/methodology" },
          ],
        },
        { text: "FAQ", link: "/faq" },
      ],
    },
    socialLinks: [
      { icon: "github", link: "https://github.com/unnamedlab/BikoDB" },
    ],
    footer: {
      message: "Released under the MIT License.",
      copyright: "Copyright © 2025-present unnamedlab",
    },
  },
});
