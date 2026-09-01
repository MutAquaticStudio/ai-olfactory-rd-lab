import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import type { ReactNode } from 'react';

export interface AnimatedListItem {
  id: string;
  content: ReactNode;
}

export default function AnimatedList({ items, className = '' }: { items: AnimatedListItem[]; className?: string }) {
  const reduced = useReducedMotion();
  return (
    <div className={className} role="log" aria-live="polite">
      <AnimatePresence initial={false}>
        {items.map((item) => (
          <motion.div
            key={item.id}
            initial={reduced ? false : { opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.18 }}
          >
            {item.content}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
