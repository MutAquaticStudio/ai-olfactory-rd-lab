import type { HTMLAttributes, ReactNode } from 'react';

export default function Panel({ title, actions, children, className = '', ...props }: HTMLAttributes<HTMLElement> & {
  title?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className={`panel ${className}`} {...props}>
      {(title || actions) && <header className="panel-header"><h2>{title}</h2>{actions}</header>}
      {children}
    </section>
  );
}
