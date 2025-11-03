document.addEventListener("DOMContentLoaded", function () {
  const API_URL = "http://127.0.0.1:8000/api/search"; // 절대경로
  const searchBtn = document.getElementById("searchBtn");
  const queryInput = document.getElementById("queryInput");
  const resultSection = document.getElementById("resultSection");
  const toggleSqlBtn = document.getElementById("toggleSqlBtn");
  const mainContent = document.querySelector(".content");

  if (toggleSqlBtn && mainContent) {
    toggleSqlBtn.addEventListener("click", function () {
      mainContent.classList.toggle("sql-view-active");
      toggleSqlBtn.textContent = mainContent.classList.contains("sql-view-active")
        ? "SQL 닫기" : "SQL 편집";
    });
  }

  async function safeFetchJson(url, options) {
    const res = await fetch(url, options);
    const ct = res.headers.get("content-type") || "";
    const text = await res.text();  // 항상 본문 확보
    let data = null;
    if (ct.includes("application/json")) {
      try { data = JSON.parse(text); } catch (_) { /* noop */ }
    }
    return { ok: res.ok, status: res.status, data, text, ct };
  }

  if (searchBtn) {
    searchBtn.addEventListener("click", async () => {
      const query = queryInput.value.trim();
      if (!query) {
        alert("질문을 입력해주세요!");
        return;
      }

      resultSection.innerHTML = "<p>⏳ 검색 중...</p>";

      try {
        const { ok, status, data, text, ct } = await safeFetchJson(API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        });

        if (!ok) {
          resultSection.innerHTML =
            `<div style="color:#c00;">
               ⚠ 서버 오류 (${status})<br>
               <pre style="white-space:pre-wrap">${text.slice(0, 1000)}</pre>
             </div>`;
          return;
        }
        if (!data) {
          resultSection.innerHTML =
            `<div style="color:#c00;">
               ⚠ 서버가 JSON을 반환하지 않았습니다 (content-type: ${ct})<br>
               <pre style="white-space:pre-wrap">${text.slice(0, 1000)}</pre>
             </div>`;
          return;
        }
        if (data.error) {
          resultSection.innerHTML = `<div style="color:#c00;">⚠ ${data.error}</div>`;
          return;
        }

        const sqlBlock = `<pre>${data.sql_text || "SQL 없음"}</pre>`;
        const summary = `
          <p>
            '${data.last_query}' → ${data.count}개 결과<br>
            ${data.opinion ? `오피니언: "${data.opinion}" (${data.main}/${data.sub})` : ""}
          </p>`;

        const table = (data.rows && data.rows.length)
          ? `<table class="result-table">
               <thead><tr>${data.columns.map(c=>`<th>${c}</th>`).join("")}</tr></thead>
               <tbody>${data.rows.slice(0,10)
                 .map(r=>`<tr>${r.map(v=>`<td>${v}</td>`).join("")}</tr>`).join("")}</tbody>
             </table>`
          : `<p style="text-align:center;">검색 결과가 없습니다.</p>`;

        resultSection.innerHTML = `
          <div class="sql-card">${sqlBlock}</div>
          <div class="result-summary">${summary}</div>
          ${table}
        `;
      } catch (err) {
        resultSection.innerHTML = `<div style="color:#c00;">⚠ 요청 실패: ${err}</div>`;
      }
    });
  }
});
