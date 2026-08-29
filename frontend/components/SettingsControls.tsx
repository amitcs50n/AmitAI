"use client";

import { cn } from "@/lib/utils";

export function SettingsSection({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-b border-[#4d392b]/45 py-7 first:pt-0 last:border-b-0">
      <h2 className="font-serif text-2xl text-[#eee8e1]">{title}</h2>
      {description ? <p className="mt-1 text-sm leading-6 text-[#8c857f]">{description}</p> : null}
      <div className="mt-4 overflow-hidden rounded-xl border border-[#5d4533]/50 bg-[#111211]">{children}</div>
    </section>
  );
}

export function SettingRow({
  label,
  description,
  value,
}: {
  label: string;
  description?: string;
  value?: React.ReactNode;
}) {
  return (
    <div className="flex min-h-16 items-center justify-between gap-5 border-b border-[#453429]/45 px-4 py-3 last:border-b-0 sm:px-5">
      <div>
        <div className="text-sm text-[#ddd6d0]">{label}</div>
        {description ? <p className="mt-0.5 text-xs leading-5 text-[#817a74]">{description}</p> : null}
      </div>
      {value ? <div className="shrink-0 text-sm text-[#a69e97]">{value}</div> : null}
    </div>
  );
}

export function Toggle({
  checked,
  label,
  onChange,
  disabled = false,
}: {
  checked: boolean;
  label: string;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      aria-checked={checked}
      aria-label={label}
      className={cn(
        "relative h-6 w-11 rounded-full border transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254] focus-visible:ring-offset-2 focus-visible:ring-offset-[#111211] disabled:cursor-not-allowed disabled:opacity-50",
        checked ? "border-[#b4784b] bg-[#8e5e3c]" : "border-[#5f544c] bg-[#292725]",
      )}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      role="switch"
      type="button"
    >
      <span
        aria-hidden="true"
        className={cn(
          "absolute top-0.5 h-4.5 w-4.5 rounded-full bg-[#eee8e1] transition-transform",
          checked ? "translate-x-[1.28rem]" : "translate-x-0.5",
        )}
      />
    </button>
  );
}
