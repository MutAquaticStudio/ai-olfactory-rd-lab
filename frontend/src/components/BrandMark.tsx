export default function BrandMark({ size = 34 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <path d="M24 3 34 9v12l-10 6-10-6V9L24 3Z" stroke="currentColor" strokeWidth="2" />
      <path d="m14 21-9 5v12l10 6 9-5V27M34 21l9 5v12l-10 6-9-5" stroke="currentColor" strokeWidth="2" />
      <circle cx="24" cy="15" r="2.6" fill="currentColor" />
      <circle cx="14.8" cy="32.5" r="2.6" fill="currentColor" />
      <circle cx="33.2" cy="32.5" r="2.6" fill="currentColor" />
    </svg>
  );
}
