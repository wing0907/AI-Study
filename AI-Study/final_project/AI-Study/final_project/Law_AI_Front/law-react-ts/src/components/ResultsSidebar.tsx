// src/components/ResultsSidebar.tsx
type Facet = {
  id: string;
  title: string;
  items: string[];
};

export default function ResultsSidebar({
  facets,
  active,
  onPick,
}: {
  facets: Facet[];
  active: string | null;
  onPick: (facetId: string) => void;
}) {
  return (
    <aside className="resultsSidebar">
      <div className="sidebarTitle">키워드</div>
      {facets.map((f) => (
        <button
          key={f.id}
          className={`facetBtn ${active === f.id ? 'active' : ''}`}
          onClick={() => onPick(f.id)}
        >
          <span className="facetTitle">{f.title}</span>
          <span className="facetCount">{f.items.length}</span>
        </button>
      ))}
    </aside>
  );
}
