/**
 * Client-Side Legal Document Generator
 * Formats statutory appeals, contract redlines, and HIPAA requests with formal legal margins and citations.
 */

export interface LegalDocumentOptions {
  title: string;
  patientName: string;
  policyId?: string;
  claimId?: string;
  statutoryCitations: string[];
  bodyText: string;
}

export function generateLegalDocumentHtml(options: LegalDocumentOptions): string {
  const citationsList = options.statutoryCitations
    .map((cite, i) => `<li><sup>[${i + 1}]</sup> ${cite}</li>`)
    .join("\n");

  return `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>${options.title}</title>
      <style>
        @page {
          size: letter;
          margin: 1in;
        }
        body {
          font-family: "Times New Roman", Times, serif;
          font-size: 12pt;
          line-height: 1.5;
          color: #000;
          padding: 1in;
          max-width: 800px;
          margin: 0 auto;
        }
        .header {
          text-align: center;
          font-weight: bold;
          font-size: 14pt;
          text-transform: uppercase;
          margin-bottom: 24pt;
          border-bottom: 2px solid #000;
          padding-bottom: 8pt;
        }
        .meta-table {
          width: 100%;
          border-collapse: collapse;
          margin-bottom: 18pt;
        }
        .meta-table td {
          padding: 3pt 0;
          vertical-align: top;
        }
        .meta-label {
          font-weight: bold;
          width: 160px;
        }
        .body-content {
          white-space: pre-wrap;
          text-align: justify;
          margin-bottom: 30pt;
        }
        .footnotes {
          border-top: 1px solid #666;
          padding-top: 10pt;
          margin-top: 40pt;
          font-size: 10pt;
          color: #333;
        }
        .footnotes ul {
          list-style: none;
          padding-left: 0;
        }
      </style>
    </head>
    <body>
      <div class="header">${options.title}</div>
      <table class="meta-table">
        <tr><td class="meta-label">PATIENT NAME:</td><td>${options.patientName}</td></tr>
        ${options.policyId ? `<tr><td class="meta-label">POLICY / MEMBER ID:</td><td>${options.policyId}</td></tr>` : ""}
        ${options.claimId ? `<tr><td class="meta-label">CLAIM / CASE ID:</td><td>${options.claimId}</td></tr>` : ""}
        <tr><td class="meta-label">DATE:</td><td>${new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}</td></tr>
      </table>
      
      <div class="body-content">${options.bodyText}</div>

      ${options.statutoryCitations.length > 0 ? `
        <div class="footnotes">
          <strong>STATUTORY CITATIONS & GOVERNING LAW:</strong>
          <ul>${citationsList}</ul>
        </div>
      ` : ""}
    </body>
    </html>
  `;
}
