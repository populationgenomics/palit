import "@mantine/core/styles.css";
import "@flowajs/react-viewer/styles.css";

import {
  PdfHighlightViewer,
  type HighlightBbox,
  type PdfHighlight,
} from "@flowajs/react-viewer";
import {
  Alert,
  Anchor,
  Box,
  Loader,
  MantineProvider,
  ScrollArea,
  Stack,
  Text,
  Title,
  UnstyledButton,
} from "@mantine/core";
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

/** One paper's quotes, written by `palit generate-report` to citations/<paper>.json. */
interface PaperCitations {
  doi: string;
  title: string;
  quotes: { quote: string; bboxes: HighlightBbox[] }[];
}

// The page lives in report_X/viewer/; papers and citation files are siblings.
// `paper` is the PDF's file stem (the percent-encoded DOI), so it is encoded
// once more to form the URL path of a file whose name contains '%'.
const citationsUrl = (paper: string) => `../citations/${encodeURIComponent(paper)}.json`;
const pdfUrl = (paper: string) => `../papers/${encodeURIComponent(paper)}.pdf`;

function firstPage(bboxes: HighlightBbox[]): number | null {
  return bboxes.length > 0 ? bboxes[0].page : null;
}

function Viewer({ paper, initialQuote }: { paper: string; initialQuote: number | null }) {
  const [citations, setCitations] = useState<PaperCitations | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<number | null>(initialQuote);

  useEffect(() => {
    fetch(citationsUrl(paper))
      .then((response) => {
        if (!response.ok) throw new Error(`${response.status} loading ${citationsUrl(paper)}`);
        return response.json() as Promise<PaperCitations>;
      })
      .then(setCitations)
      .catch((e: unknown) => setError(String(e)));
  }, [paper]);

  if (error !== null) return <Alert color="red" title="Could not load this paper">{error}</Alert>;
  if (citations === null) return <Loader m="xl" />;

  const selected = active !== null ? citations.quotes[active] : undefined;
  const highlights: PdfHighlight[] = selected
    ? [{ bboxes: selected.bboxes, label: selected.quote }]
    : [];

  return (
    <Box style={{ display: "flex", height: "100vh" }}>
      <Stack w={380} p="md" gap="sm" style={{ borderRight: "1px solid #ddd", flexShrink: 0 }}>
        <Title order={5}>{citations.title}</Title>
        <Anchor href={`https://doi.org/${citations.doi}`} target="_blank" size="sm">
          doi:{citations.doi}
        </Anchor>
        <Text size="xs" c="dimmed">
          Quotes cited in the report ({citations.quotes.length}); click one to highlight it.
        </Text>
        <ScrollArea style={{ flex: 1 }}>
          <Stack gap={6}>
            {citations.quotes.map((entry, index) => {
              const page = firstPage(entry.bboxes);
              return (
                <UnstyledButton
                  key={index}
                  onClick={() => setActive(index)}
                  p={6}
                  style={{
                    borderRadius: 4,
                    background: index === active ? "rgb(255, 236, 179)" : undefined,
                  }}
                >
                  <Text size="sm">{entry.quote}</Text>
                  <Text size="xs" c="dimmed">
                    {page !== null ? `page ${page}` : "not located in the PDF"}
                  </Text>
                </UnstyledButton>
              );
            })}
          </Stack>
        </ScrollArea>
      </Stack>
      <Box style={{ flex: 1, minWidth: 0 }}>
        <PdfHighlightViewer
          pdfUrl={pdfUrl(paper)}
          highlights={highlights}
          workerSrc="./pdfjs/pdf.worker.min.mjs"
          cMapUrl="./pdfjs/cmaps/"
        />
      </Box>
    </Box>
  );
}

function App() {
  const params = new URLSearchParams(window.location.search);
  const paper = params.get("paper");
  const quote = params.get("q");
  if (paper === null) {
    return <Alert color="red" title="No paper selected">Open this page from a report link.</Alert>;
  }
  return <Viewer paper={paper} initialQuote={quote === null ? null : Number(quote)} />;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <MantineProvider>
      <App />
    </MantineProvider>
  </StrictMode>,
);
