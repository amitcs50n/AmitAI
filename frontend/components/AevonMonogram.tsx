import { cn } from "@/lib/utils";

interface AevonMonogramProps {
  className?: string;
}

export function AevonMonogram({ className }: AevonMonogramProps) {
  return (
    <div
      aria-hidden="true"
      className={cn(
        "flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-[#a8754c]/65 bg-[#11100f] font-serif text-[1.55rem] leading-none text-[#d29a6a] shadow-[inset_0_0_18px_rgba(157,100,57,0.05)]",
        className,
      )}
    >
      Ae
    </div>
  );
}
