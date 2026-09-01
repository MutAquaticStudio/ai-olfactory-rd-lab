import { useRef, type MouseEventHandler, type PropsWithChildren } from 'react';

export default function SpotlightCard({ children, className = '' }: PropsWithChildren<{ className?: string }>) {
  const ref = useRef<HTMLDivElement>(null);
  const onMouseMove: MouseEventHandler<HTMLDivElement> = (event) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect || !ref.current) return;
    ref.current.style.setProperty('--mouse-x', `${event.clientX - rect.left}px`);
    ref.current.style.setProperty('--mouse-y', `${event.clientY - rect.top}px`);
  };
  return <div ref={ref} onMouseMove={onMouseMove} className={`spotlight-card ${className}`}>{children}</div>;
}
