<script setup lang="ts">
const props = withDefaults(defineProps<{
  size?: number
  strokeWidth?: number
  color?: string
}>(), {
  size: 16,
  strokeWidth: 2,
  color: 'currentColor',
})
</script>

<template>
  <svg
    aria-hidden="true"
    focusable="false"
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    :stroke="color"
    :stroke-width="String(strokeWidth)"
    stroke-linecap="round"
    stroke-linejoin="round"
  >
    <slot />
  </svg>
</template>
