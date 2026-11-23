<script setup lang="ts">
import LucideIcon from './LucideIcon.vue'
</script>

<template>
  <LucideIcon v-bind="$props">
    <line x1="12" y1="5" x2="12" y2="19" />
    <line x1="5" y1="12" x2="19" y2="12" />
  </LucideIcon>
</template>
