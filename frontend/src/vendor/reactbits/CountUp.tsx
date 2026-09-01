import { useInView, useMotionValue, useSpring } from 'motion/react';
import { useEffect, useRef } from 'react';

export default function CountUp({ to, decimals = 0 }: { to: number; decimals?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const value = useMotionValue(0);
  const spring = useSpring(value, { damping: 35, stiffness: 140 });
  const visible = useInView(ref, { once: true });
  useEffect(() => { if (visible) value.set(to); }, [to, value, visible]);
  useEffect(() => spring.on('change', (latest) => {
    if (ref.current) ref.current.textContent = latest.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  }), [decimals, spring]);
  return <span ref={ref}>0</span>;
}
